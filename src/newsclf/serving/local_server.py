"""Local, TF-Serving-compatible REST server (FastAPI).

Use it when TF Serving is not available (e.g. Windows without Docker). It
serves the same SavedModel version folders and honours the same URL scheme:

    GET  /v1/models/{name}
    GET  /v1/models/{name}/metadata
    POST /v1/models/{name}:predict
    POST /v1/models/{name}/versions/{v}:predict
    POST /v1/models/{name}/labels/{label}:predict

It also exposes the product-facing endpoint the news app actually calls:

    POST /v1/classify   {"titles": [...], "label": "stable"}

which returns topics plus the human-in-the-loop routing decision and the
language/OOV guard, i.e. the full online inference path.
"""

from __future__ import annotations

import re
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from newsclf.deployment.routing import RoutingPolicy, route_predictions
from newsclf.monitoring.drift import InputProfiler
from newsclf.serving.predictor import LocalBackend, TopicClassifier
from newsclf.serving.registry import Registry
from newsclf.utils import get_logger

log = get_logger(__name__)
_PATH = re.compile(
    r"^(?P<name>[^/:]+)(?:/versions/(?P<version>\d+)|/labels/(?P<label>[^/:]+))?(?P<verb>:predict|/metadata)?$"
)


class ClassifyRequest(BaseModel):
    titles: list[str] = Field(..., min_length=1, max_length=1000)
    label: str = "stable"


def create_app(cfg: dict) -> FastAPI:
    app = FastAPI(title="News Topic Classifier", version="1.0.0")
    state = {"registry": Registry(cfg), "requests": 0, "started": time.time()}

    def registry() -> Registry:
        # Re-read on every call so `newsclf promote` takes effect without a restart,
        # mirroring TF Serving's config polling.
        reg = Registry(cfg)
        if reg.data != state["registry"].data:
            state["registry"] = reg
            state.pop("backend", None)
        return state["registry"]

    def backend() -> LocalBackend:
        if "backend" not in state:
            state["backend"] = LocalBackend(registry())
        return state["backend"]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "uptime_s": round(time.time() - state["started"], 1),
            "requests": state["requests"],
            "labels": registry().data["labels"],
        }

    @app.get("/v1/models/{path:path}")
    def model_status(path: str):
        m = _PATH.match(path)
        reg = registry()
        if not m or m["name"] != reg.model_name:
            raise HTTPException(404, f"Servable not found for request: {path}")
        if m["verb"] == "/metadata":
            v = reg.version_for(m["label"], int(m["version"]) if m["version"] else None)
            return {"model_spec": {"name": reg.model_name, "version": str(v)}, "metadata": reg.info(v)}
        return {
            "model_version_status": [
                {"version": v, "state": "AVAILABLE", "status": {"error_code": "OK", "error_message": ""}}
                for v in sorted(reg.data["versions"], key=int, reverse=True)
            ]
        }

    @app.post("/v1/models/{path:path}")
    async def predict(path: str, request: Request):
        m = _PATH.match(path)
        reg = registry()
        if not m or m["verb"] != ":predict" or m["name"] != reg.model_name:
            return JSONResponse({"error": f"Malformed request: POST /v1/models/{path}"}, status_code=404)
        body = await request.json()
        instances = body.get("instances", body.get("inputs"))
        if instances is None:
            return JSONResponse({"error": "Missing 'instances' key"}, status_code=400)
        try:
            preds = backend().predict(instances, version=int(m["version"]) if m["version"] else None, label=m["label"])
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": f"Prediction failed: {exc}"}, status_code=400)
        state["requests"] += 1
        return {"predictions": preds}

    @app.post("/v1/classify")
    def classify(req: ClassifyRequest):
        reg = registry()
        try:
            version = reg.version_for(req.label)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        info = reg.info(version)
        clf = TopicClassifier(reg, backend())
        probs, labels, _ = clf.predict_proba(req.titles, version=version)
        profiler = InputProfiler.for_version(cfg, info, reg.version_dir(version))
        profile = profiler.profile(req.titles)
        policy = RoutingPolicy.from_version(cfg, info)
        decisions = route_predictions(req.titles, probs, labels, policy, profile)
        state["requests"] += 1
        return {
            "model_version": version,
            "label": req.label,
            "policy": policy.as_dict(),
            "results": decisions.replace({np.nan: None}).to_dict(orient="records"),
        }

    return app

import csv, glob, io, os, pickle, sys, time, types, zipfile
import cv2
import numpy as np
import numpy.random._pickle as np_pickle
import torch
import torch.nn as nn

sys.path.insert(0, "/home/nano/ultralytics_src")

class _DummyObj:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, s): pass
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _DummyObj()
    def __call__(self, *a, **kw):
        return a[0] if a else None

class _SafeRandomState(np.random.RandomState):
    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except Exception:
            pass

class _StubMod(types.ModuleType):
    def __init__(self, n):
        super().__init__(n)
        self.__path__ = []
    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"): raise AttributeError(item)
        if self.__name__.startswith("numpy._core"):
            try:
                m = __import__(self.__name__.replace("numpy._core", "numpy.core", 1), fromlist=[item])
                if hasattr(m, item): return getattr(m, item)
            except Exception: pass
        return _DummyObj

class _CompatFinder:
    def find_module(self, fullname, path=None):
        return self if (fullname == "albumentations" or fullname.startswith(("albumentations.", "numpy._core"))) else None
    def load_module(self, fullname):
        return sys.modules.setdefault(fullname, _StubMod(fullname))

sys.meta_path.insert(0, _CompatFinder())

import numpy.core as np_core
sys.modules["numpy._core"] = np_core
for attr in dir(np_core):
    if isinstance(getattr(np_core, attr), types.ModuleType):
        sys.modules["numpy._core." + attr] = getattr(np_core, attr)

_orig_ctor = np_pickle.__bit_generator_ctor
np_pickle.__bit_generator_ctor = lambda name="MT19937": _DummyObj() if "SeedSequence" in str(name) else _orig_ctor("MT19937" if "MT19937" not in str(name) else name)

import ultralytics.nn.modules as nn_mods
import ultralytics.yolo.utils as yolo_utils
import ultralytics.yolo.utils.torch_utils as yolo_torch_utils
sys.modules["ultralytics.utils"] = yolo_utils
sys.modules["ultralytics.utils.torch_utils"] = yolo_torch_utils
for attr in dir(yolo_utils):
    if hasattr(getattr(yolo_utils, attr), "__name__"):
        sys.modules["ultralytics.utils." + attr] = getattr(yolo_utils, attr)

class _CompatBlock(nn.Module):
    def __init__(self, *a, **kw):
        super().__init__()
        self._adapt = None
    def forward(self, x):
        adapt = getattr(self, "_adapt", None)
        cv2 = getattr(self, "cv2", None)
        if isinstance(cv2, nn.Module):
            out_c = getattr(getattr(cv2, "conv", cv2), "out_channels", x.shape[1])
        else:
            out_c = x.shape[1]
        if x.shape[1] != out_c:
            if adapt is None or adapt.in_channels != x.shape[1] or adapt.out_channels != out_c:
                adapt = nn.Conv2d(x.shape[1], out_c, 1, bias=False).to(x.device)
                self._adapt = adapt
            return adapt(x)
        return x

for name in ("C3k", "C3k2", "A2C2f", "C2PSA", "PSABlock", "Attention"):
    if not hasattr(nn_mods, name): setattr(nn_mods, name, _CompatBlock)
for sub in ("conv", "block", "head", "utils", "tal"):
    sys.modules["ultralytics.nn.modules." + sub] = nn_mods

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if "SeedSequence" in name or "SeedSequence" in mod or "BitGenerator" in name:
            return _DummyObj
        if name == "RandomState":
            return _SafeRandomState
        return super().find_class(mod, name)

_NPDTYPE = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "ByteStorage": np.uint8, "CharStorage": np.int8, "ShortStorage": np.int16,
    "IntStorage": np.int32, "LongStorage": np.int64,
}

def load_any(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with zipfile.ZipFile(path, "r") as zf:
            pkl_name = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"][0]
            prefix = pkl_name[:-len("data.pkl")]
            cache = {}
            def pload(sid):
                if isinstance(sid, tuple) and len(sid) >= 5 and sid[0] == "storage":
                    _, st, key, _, _ = sid[:5]
                    sname = st.__name__ if hasattr(st, "__name__") else str(st)
                    k = (sname, str(key))
                    if k not in cache:
                        np_dt = _NPDTYPE.get(sname, np.uint8)
                        arr = np.frombuffer(zf.read(prefix + "data/" + str(key)), dtype=np_dt).copy()
                        cache[k] = torch.from_numpy(arr).storage()
                    return cache[k]
                return sid
            up = _SafeUnpickler(io.BytesIO(zf.read(pkl_name)))
            up.persistent_load = pload
            return up.load()

def nms_torch(boxes, scores, iou_thresh=0.45):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1: break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long)

def decode_valid(out, names, orig_w, orig_h, conf_thresh=0.25, iou_thresh=0.45):
    pred = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(pred, torch.Tensor) or pred.ndim != 3:
        return []
    p = pred[0]
    if p.shape[0] < p.shape[1]:
        p = p.transpose(0, 1)
    boxes_xywh = p[:, :4]
    cls_scores = p[:, 4:]
    if cls_scores.min() < 0 or cls_scores.max() > 1:
        cls_scores = cls_scores.sigmoid()
    max_scores, cls_ids = cls_scores.max(dim=1)
    keep = max_scores >= conf_thresh
    if keep.sum() == 0:
        return []
    boxes_xywh, max_scores, cls_ids = boxes_xywh[keep], max_scores[keep], cls_ids[keep]

    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1_640, y1_640 = cx - bw / 2.0, cy - bh / 2.0
    x2_640, y2_640 = cx + bw / 2.0, cy + bh / 2.0
    idx_keep = nms_torch(torch.stack([x1_640, y1_640, x2_640, y2_640], dim=1), max_scores, iou_thresh)

    valid = []
    for i in idx_keep[:20]:
        bx1 = int(max(0, round(x1_640[i].item() * orig_w / 640.0)))
        by1 = int(max(0, round(y1_640[i].item() * orig_h / 640.0)))
        bx2 = int(min(orig_w - 1, round(x2_640[i].item() * orig_w / 640.0)))
        by2 = int(min(orig_h - 1, round(y2_640[i].item() * orig_h / 640.0)))
        if (bx2 - bx1) < 2 or (by2 - by1) < 2:
            continue
        cid = int(cls_ids[i].item())
        sc = float(max_scores[i].item())
        label = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
        valid.append((label, sc, (bx1, by1, bx2, by2)))
    return valid

in_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/nano/single_test"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/nano/batch_out"
os.makedirs(out_dir, exist_ok=True)

ckpts = [
    ("train_10class", "/home/nano/pi_pcb_extracted/home/ece/bharat_ai_ps5/runs/detect/train_10class/weights/best.pt", (0, 255, 0)),
    ("pretrained_pcb", "/home/nano/pi_pcb_extracted/home/ece/bharat_ai_ps5/pretrained_pcb_model/best.pt", (0, 200, 255)),
]

models = []
for tag, p, color in ckpts:
    try:
        ck = load_any(p)
        m = (ck.get("model") or ck.get("ema")).float().eval()
        names = getattr(m, "names", {})
        models.append((tag, m, names, color))
        print("Loaded %s" % tag)
    except Exception as e:
        print("Skip %s: %s: %s" % (tag, type(e).__name__, e))

assert models, "No model loaded"

exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
img_paths = []
for ext in exts:
    img_paths.extend(glob.glob(os.path.join(in_dir, ext)))
    img_paths.extend(glob.glob(os.path.join(in_dir, ext.upper())))
img_paths = sorted(set(p for p in img_paths if not os.path.basename(p).startswith("out_")))
print("Found %d images in %s" % (len(img_paths), in_dir))

csv_path = os.path.join(out_dir, "detections.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "model", "label", "confidence", "x1", "y1", "x2", "y2", "infer_ms"])

    for idx, ip in enumerate(img_paths, 1):
        img = cv2.imread(ip)
        if img is None:
            print("[%d/%d] Unreadable: %s" % (idx, len(img_paths), ip))
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        vis = img.copy()
        base = os.path.basename(ip)
        total_dets = 0

        for tag, m, names, color in models:
            t0 = time.time()
            with torch.no_grad():
                out = m(x)
            ms = (time.time() - t0) * 1000.0
            dets = decode_valid(out, names, w, h, conf_thresh=0.25, iou_thresh=0.45)
            total_dets += len(dets)
            for label, sc, (x1, y1, x2, y2) in dets:
                writer.writerow([base, tag, label, "%.4f" % sc, x1, y1, x2, y2, "%.1f" % ms])
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, "%s %.2f" % (label, sc), (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = os.path.join(out_dir, "out_" + base)
        cv2.imwrite(out_path, vis)
        print("[%d/%d] %s -> %d dets -> %s" % (idx, len(img_paths), base, total_dets, out_path))

print("Done. Annotated images + CSV saved in %s" % out_dir)

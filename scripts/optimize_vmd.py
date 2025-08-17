# VMD Motion Optimizer by Barış Keser (barkeser2002)
# License: GNU General Public License v3.0 (GPL-3.0)
# See LICENSE for details.

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional

import numpy as np
from tqdm import tqdm

import os
import tempfile
import io
import struct

# ---------- Minimal VMD IO (v2) ----------

CP932 = 'cp932'


def _read_u32(f: io.BufferedReader) -> int:
    return struct.unpack('<I', f.read(4))[0]


def _read_f32(f: io.BufferedReader, n: int = 1):
    return struct.unpack('<' + 'f' * n, f.read(4 * n))


def _read_name(f: io.BufferedReader, size: int) -> str:
    raw = f.read(size)
    if not raw:
        return ''
    if b'\x00' in raw:
        raw = raw.split(b'\x00', 1)[0]
    try:
        return raw.decode(CP932, errors='ignore')
    except Exception:
        return raw.decode('latin1', errors='ignore')


def _write_name(f: io.BufferedWriter, name: str, size: int):
    try:
        data = name.encode(CP932, errors='ignore')
    except Exception:
        data = name.encode('latin1', errors='ignore')
    data = (data + b'\x00' * size)[:size]
    f.write(data)


@dataclass
class BoneFrame:
    name: str
    frame: int
    pos: np.ndarray  # shape (3,)
    quat: np.ndarray  # shape (4,) (x,y,z,w)
    interp: bytes | None = None  # 64 bytes


@dataclass
class MorphFrame:
    name: str
    frame: int
    weight: float


@dataclass
class Motion:
    model_name: str
    bones: List[BoneFrame]
    morphs: List[MorphFrame]


# Yardımcılar: seri yumuşatma ve derinlik (Z) hizası kaldırma

def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) == 0:
        return values
    w = min(window, len(values))
    arr = np.array(values, dtype=np.float64)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    out = (csum[w:] - csum[:-w]) / float(w)
    # kenar doldurma: baş/sonu en yakın ortalama ile doldur
    head = [out[0]] * (w - 1)
    tail = []
    if len(out) < len(arr):
        tail = [out[-1]] * (len(arr) - len(out))
    return list(head) + list(out) + tail


def remove_depth_alignment(motion: Motion,
                           root_candidates: Optional[List[str]] = None,
                           smooth_window: int = 0,
                           scale: float = 1.0) -> None:
    """
    Global derinlik (Z) ötelemesini kök kemikten ölçer ve TÜM kemiklere uygular.
    - root_candidates: ['センター','Center','センタ','全ての親','AllParent'] gibi isimlerle kök kemik aranır.
    - smooth_window: kök Z serisine hareketli ortalama (frame cinsinden) uygular.
    - scale: çıkarılacak Z ofseti için çarpan.
    İşlem doğrudan motion.bones üzerinde değişiklik yapar (global çeviri sağlar).
    """
    if root_candidates is None:
        # Daha kapsamlı kök aday listesi
        root_candidates = [
            '全ての親', 'AllParent',
            'センター', 'Center', 'センタ',
            'グルーブ', 'Groove',
            'Root', 'root'
        ]

    # kök kemik adını seç
    bone_names = set(b.name for b in motion.bones)
    root_name = None
    for cand in root_candidates:
        if cand in bone_names:
            root_name = cand
            break
    if root_name is None:
        return  # kök bulunamadı; işlem yok

    # root Z serisini topla (frame->z)
    frames = []
    zs = []
    for b in motion.bones:
        if b.name == root_name:
            frames.append(int(b.frame))
            zs.append(float(b.pos[2]))
    if not frames:
        return

    # frame bazlı z listelerini sırala ve eşleştir
    order = np.argsort(frames)
    frames = [frames[i] for i in order]
    zs = [zs[i] for i in order]

    # yumuşatma uygula
    if smooth_window and smooth_window > 1:
        zs = _moving_average(zs, smooth_window)

    # hızlı arama için
    frame_arr = np.array(frames, dtype=np.int64)
    z_arr = np.array(zs, dtype=np.float64)

    def z_at(f: int) -> float:
        # tam eşleşme
        idx = np.searchsorted(frame_arr, f)
        if idx < len(frame_arr) and frame_arr[idx] == f:
            return float(z_arr[idx]) * scale
        # interpolasyon
        i1 = idx - 1
        i2 = idx
        if i1 < 0:
            return float(z_arr[0]) * scale
        if i2 >= len(frame_arr):
            return float(z_arr[-1]) * scale
        f1, f2 = frame_arr[i1], frame_arr[i2]
        z1, z2 = z_arr[i1], z_arr[i2]
        if f2 == f1:
            return float(z1) * scale
        t = (f - f1) / float(f2 - f1)
        return float(z1 * (1 - t) + z2 * t) * scale

    # TÜM kemik keylerine bu çerçeveye ait Z ofsetini uygula (global çeviri)
    for b in motion.bones:
        zoff = z_at(int(b.frame))
        b.pos[2] = float(b.pos[2]) - zoff


def stabilize_ground(motion: Motion,
                     target_y: float = 0.0,
                     use_feet_only: bool = True,
                     feet_candidates: Optional[List[str]] = None,
                     smooth_window: int = 0,
                     scale: float = 1.0,
                     root_candidates: Optional[List[str]] = None) -> None:
    """
    Ground stabilization: Her framedeki en düşük Y değerini (varsayılan ayak kemikleri) bulup,
    karakteri hedef zemine (target_y) oturtur. Ofseti TÜM kemiklere uygular (global çeviri),
    böylece IK/ayak sabitken gövdenin "yukarı doğru uzaması" engellenir.
    """
    if feet_candidates is None:
        feet_candidates = [
            '左足', '右足', '左足ＩＫ', '右足ＩＫ', '左足IK', '右足IK', 'つま先', 'つま先ＩＫ',
            'Toe', 'ToeIK', 'Ankle', 'Foot', 'LeftFoot', 'RightFoot', 'LeftAnkle', 'RightAnkle',
            '足', '足ＩＫ', '足IK', '足首'
        ]

    # kök adayları (ölçüm için gerekebilir ama artık uygulama tüm kemiklere)
    if root_candidates is None:
        root_candidates = [
            '全ての親', 'AllParent',
            'センター', 'Center', 'センタ',
            'グルーブ', 'Groove',
            'Root', 'root'
        ]

    names = set(b.name for b in motion.bones)

    # Ayak/kemik seçim seti (minY ölçümü için)
    if use_feet_only:
        selected = [n for n in names for cand in feet_candidates if cand in n]
        selected = set(selected)
        if not selected:
            selected = names  # fallback: tüm kemikler
    else:
        selected = names

    # frame -> minY haritası (seçili kemikler arasında)
    minY: Dict[int, float] = {}
    for b in motion.bones:
        if b.name not in selected:
            continue
        f = int(b.frame)
        y = float(b.pos[1])
        if f not in minY or y < minY[f]:
            minY[f] = y
    if not minY:
        return

    frames = sorted(minY.keys())
    lows = [minY[f] for f in frames]
    if smooth_window and smooth_window > 1:
        lows = _moving_average(lows, smooth_window)

    farr = np.array(frames, dtype=np.int64)
    yarr = np.array(lows, dtype=np.float64)

    def y_off_at(f: int) -> float:
        # minY(f) - target_y
        idx = np.searchsorted(farr, f)
        if idx < len(farr) and farr[idx] == f:
            base = float(yarr[idx])
        else:
            i1 = idx - 1
            i2 = idx
            if i1 < 0:
                base = float(yarr[0])
            elif i2 >= len(farr):
                base = float(yarr[-1])
            else:
                f1, f2 = farr[i1], farr[i2]
                y1, y2 = yarr[i1], yarr[i2]
                if f2 == f1:
                    base = float(y1)
                else:
                    t = (f - f1) / float(f2 - f1)
                    base = float(y1 * (1 - t) + y2 * t)
        return (base - target_y) * scale

    # Ofseti TÜM kemik keylerine uygula (global yükseltme/alçaltma)
    for b in motion.bones:
        off = y_off_at(int(b.frame))
        b.pos[1] = float(b.pos[1]) - off


def read_vmd(path: str) -> Motion | None:
    with open(path, 'rb') as f:
        data = f.read()
    if not (data.startswith(b"Vocaloid Motion Data 0002") or data.startswith(b"Vocaloid Motion Data file")):
        print('invalid signature', data[:30])
        return None
    # header
    model_name_bytes = data[30:50]
    if b"\x00" in model_name_bytes:
        model_name_bytes = model_name_bytes.split(b"\x00", 1)[0]
    try:
        model_name = model_name_bytes.decode(CP932, errors='ignore')
    except Exception:
        model_name = model_name_bytes.decode('latin1', errors='ignore')

    # standart ofsetler
    pos = 50
    total = len(data)
    def u32_at(off: int) -> int:
        if off + 4 > total:
            return -1
        return struct.unpack('<I', data[off:off+4])[0]
    def f32s_at(off: int, n: int):
        end = off + 4*n
        if end > total:
            return None
        return struct.unpack('<' + 'f'*n, data[off:end])

    bone_count = u32_at(pos)
    def try_parse(bc: int):
        off = pos + 4
        bones: List[BoneFrame] = []
        for _ in range(bc):
            if off + 15 + 4 + 7*4 + 64 > total:
                return None
            name_b = data[off:off+15]
            off += 15
            if b"\x00" in name_b:
                name_b = name_b.split(b"\x00", 1)[0]
            try:
                name = name_b.decode(CP932, errors='ignore')
            except Exception:
                name = name_b.decode('latin1', errors='ignore')
            frame = struct.unpack('<I', data[off:off+4])[0]
            off += 4
            vals = f32s_at(off, 7)
            if vals is None:
                return None
            px, py, pz, qx, qy, qz, qw = vals
            off += 28
            interp = data[off:off+64]
            off += 64
            bones.append(BoneFrame(name=name, frame=frame,
                                   pos=np.array([px, py, pz], dtype=np.float32),
                                   quat=np.array([qx, qy, qz, qw], dtype=np.float32),
                                   interp=interp))
        # morph block
        if off + 4 > total:
            return None
        mc = struct.unpack('<I', data[off:off+4])[0]
        off += 4
        morphs: List[MorphFrame] = []
        for _ in range(mc):
            if off + 15 + 4 + 4 > total:
                return None
            name_b = data[off:off+15]
            off += 15
            if b"\x00" in name_b:
                name_b = name_b.split(b"\x00", 1)[0]
            try:
                name = name_b.decode(CP932, errors='ignore')
            except Exception:
                name = name_b.decode('latin1', errors='ignore')
            frame = struct.unpack('<I', data[off:off+4])[0]
            off += 4
            (w,) = struct.unpack('<f', data[off:off+4])
            off += 4
            morphs.append(MorphFrame(name=name, frame=frame, weight=float(w)))
        # camera/light sayıları için en az 8 bayt kalmalı
        if off + 8 > total:
            return None
        # opsiyonel kontroller (çoğu dosyada 0)
        cam = struct.unpack('<I', data[off:off+4])[0]
        lig = struct.unpack('<I', data[off+4:off+8])[0]
        # Eğer cam/lig çok büyükse bu aday değil
        if cam > 100000 or lig > 100000:
            return None
        return Motion(model_name=model_name, bones=bones, morphs=morphs)

    # İlk olarak doğrudan bone_count ile dene, ama dosya uzunluğu ile uyumlu mu kontrol et
    if bone_count >= 0:
        max_possible = (total - (pos + 4)) // 111
        if bone_count <= max_possible:
            parsed = try_parse(bone_count)
            if parsed is not None:
                return parsed
    # Aksi halde en olası aralıkta tarama yap
    nmax = (total - (pos + 4)) // 111
    for bc in range(nmax, max(nmax - 5000, 0), -1):
        parsed = try_parse(bc)
        if parsed is not None:
            return parsed
    # Son çare: küçük sayılardan dene
    for bc in range(0, min(5000, nmax + 1)):
        parsed = try_parse(bc)
        if parsed is not None:
            return parsed
    return None


def write_vmd(path: str, motion: Motion) -> bool:
    with open(path, 'wb') as f:
        # 30 bayt imza
        sig = b"Vocaloid Motion Data 0002"
        f.write(sig + b'\x00' * (30 - len(sig)))
        # 20 bayt model adı
        _write_name(f, motion.model_name or '', 20)
        # bone frames
        f.write(struct.pack('<I', len(motion.bones)))
        for b in motion.bones:
            _write_name(f, b.name or '', 15)
            f.write(struct.pack('<I', int(b.frame)))
            f.write(struct.pack('<7f', float(b.pos[0]), float(b.pos[1]), float(b.pos[2]),
                               float(b.quat[0]), float(b.quat[1]), float(b.quat[2]), float(b.quat[3])))
            interp = (b.interp or b'\x00' * 64)
            if len(interp) != 64:
                interp = (interp + b'\x00' * 64)[:64]
            f.write(interp)
        # morph frames
        f.write(struct.pack('<I', len(motion.morphs)))
        for m in motion.morphs:
            _write_name(f, m.name or '', 15)
            f.write(struct.pack('<I', int(m.frame)))
            f.write(struct.pack('<f', float(m.weight)))
        # camera, light sayıları (0)
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))
    return True


def _maybe_fix_vmd_header(src_path: str) -> str:
    """XR Animator vb. araçların eklediği fazladan null baytları kaldır.
    Standart VMD başlığı: 30 bayt imza + 20 bayt model adı.
    Bazı dosyalarda imza ile ad arasında fazladan 0x00 dolgu olabilir.
    Böyle bir durum varsa geçici bir dosyada düzeltilmiş kopya oluşturup yolunu döndür."""
    sig = b"Vocaloid Motion Data 0002"
    with open(src_path, 'rb') as f:
        head = f.read(256)
    if not head.startswith(sig):
        return src_path
    # imzadan sonra beklenen: hemen 20 bayt isim. Eğer arada çok sayıda 0x00 varsa sıkıştır.
    i = len(sig)
    # mevcut bazı dosyalarda 0x00 dolgularından sonra isim başlıyor
    j = i
    while j < len(head) and head[j] == 0:
        j += 1
    # j, ilk non-zero konum
    if j == i:
        return src_path  # zaten standart
    # j > i ve j < 30 ise arada beklenmeyen dolgu var demek; sonraki 20 baytı isim olarak almayı dene
    # Sadece j ilk 30 bayt içinde (imza pad alanında) düzeltme uygula
    if j >= 30:
        return src_path
    # j > i ise arada dolgu var demek; sonraki 20 baytı isim olarak almayı dene
    name_bytes = head[j:j+20]
    if len(name_bytes) < 1:
        return src_path
    # ismi 20 bayta pad/crop et
    name_bytes = (name_bytes + b"\x00" * 20)[:20]
    # geri kalan gövde, orijinalde j+20'den başlar
    with open(src_path, 'rb') as f:
        f.seek(j + 20)
        rest = f.read()
    fixed = sig + name_bytes + rest
    # geçici dosyaya yaz
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, os.path.basename(src_path) + ".fixed.vmd")
    with open(tmp_path, 'wb') as f:
        f.write(fixed)
    return tmp_path


# ---------- Yardimci matematik ----------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def quat_neg(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


@dataclass
class BoneKey:
    frame: int
    loc: Tuple[float, float, float]
    rot: Tuple[float, float, float, float]


@dataclass
class MorphKey:
    frame: int
    weight: float


# ---------- Özetleme/optimizasyon ----------

def simplify_curve(keys: List[Tuple[int, np.ndarray]], eps: float) -> List[Tuple[int, np.ndarray]]:
    """
    RDP benzeri anahtar azaltma. keys: (frame, valueVector)
    eps: maksimum sapma toleransı
    """
    if len(keys) <= 2:
        return keys

    frames = np.array([k for k, _ in keys], dtype=np.float64)
    values = np.stack([v for _, v in keys]).astype(np.float64)

    def recurse(idx0: int, idx1: int, keep_flags: np.ndarray):
        f0, f1 = frames[idx0], frames[idx1]
        v0, v1 = values[idx0], values[idx1]
        df = f1 - f0
        if df == 0:
            return
        # lineer interpolasyon ile max sapma
        t = (frames[idx0 + 1:idx1] - f0) / df
        interp = v0[None, :] * (1 - t)[:, None] + v1[None, :] * t[:, None]
        segment = values[idx0 + 1:idx1]
        err = np.max(np.linalg.norm(segment - interp, axis=1), initial=0.0)
        if err > eps:
            # en kötü noktayı tut ve böl
            rel_idx = int(np.argmax(np.linalg.norm(segment - interp, axis=1)))
            split = idx0 + 1 + rel_idx
            keep_flags[split] = True
            recurse(idx0, split, keep_flags)
            recurse(split, idx1, keep_flags)

    keep = np.zeros(len(keys), dtype=bool)
    keep[0] = True
    keep[-1] = True
    recurse(0, len(keys) - 1, keep)

    out = [(int(frames[i]), values[i].astype(np.float32)) for i, k in enumerate(keep) if k]
    return out


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    # q0, q1 normalize
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    d = quat_dot(q0, q1)
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta_0 = math.acos(max(min(d, 1.0), -1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - d * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize((s0 * q0) + (s1 * q1))


def simplify_quat_curve(keys: List[Tuple[int, np.ndarray]], eps_rad: float) -> List[Tuple[int, np.ndarray]]:
    if len(keys) <= 2:
        return keys

    frames = np.array([k for k, _ in keys], dtype=np.float64)
    quats = np.stack([q for _, q in keys]).astype(np.float64)

    # işaret sürekliliği
    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] = -quats[i]

    def ang_err(q, p):
        d = abs(float(np.dot(q, p)))
        d = max(min(d, 1.0), -1.0)
        return 2.0 * math.acos(d)  # radyan

    def recurse(idx0: int, idx1: int, keep_flags: np.ndarray):
        f0, f1 = frames[idx0], frames[idx1]
        q0, q1 = quats[idx0], quats[idx1]
        df = f1 - f0
        if df == 0:
            return
        ts = (frames[idx0 + 1:idx1] - f0) / df
        max_err = 0.0
        max_i = -1
        for j, t in enumerate(ts):
            q = slerp(q0, q1, float(t))
            e = ang_err(quats[idx0 + 1 + j], q)
            if e > max_err:
                max_err = e
                max_i = idx0 + 1 + j
        if max_err > eps_rad:
            keep_flags[max_i] = True
            recurse(idx0, max_i, keep_flags)
            recurse(max_i, idx1, keep_flags)

    keep = np.zeros(len(keys), dtype=bool)
    keep[0] = True
    keep[-1] = True
    recurse(0, len(keys) - 1, keep)

    out = [(int(frames[i]), quat_normalize(quats[i]).astype(np.float32)) for i, k in enumerate(keep) if k]
    return out


# ---------- VMD okuma/yazma sarıcıları ----------

def read_vmd_pymeshio_fallback(path: str):
    # Eski pymeshio okuması güvenilir olmadığı için kapatıldı.
    return None


def write_vmd_pymeshio_fallback(path: str, motion):
    return False


# ---------- Asıl optimizasyon akışı ----------

def optimize_vmd(input_path: str, output_path: str,
                 pos_eps: float = 0.05,
                 rot_eps_deg: float = 0.5,
                 morph_eps: float = 1e-3,
                 key_step: int = 1,
                 preserve_end_keys: bool = True,
                 remove_depth: bool = False,
                 depth_smooth_window: int = 0,
                 depth_scale: float = 1.0,
                 stabilize_ground_flag: bool = False,
                 ground_target_y: float = 0.0,
                 ground_use_feet_only: bool = True,
                 ground_smooth_window: int = 0,
                 ground_scale: float = 1.0,
                 replace_xr_with: Optional[str] = "Barış Keser",
                 progress: Optional[Callable[[str, int, int], None]] = None):
    """
    VMD Motion Optimizer by Barış Keser (barkeser2002)
    - pos_eps: pozisyon için max dünyasal sapma (model birimi)
    - rot_eps_deg: quaternion açısal hata eşiği (derece)
    - morph_eps: morph ağırlığı için tolerans
    - key_step: her n karede bir downsample başlangıç filtresi (opsiyonel)
    - preserve_end_keys: her kanalın ilk/son karesini koru
    """
    fixed_path = _maybe_fix_vmd_header(input_path)
    m = read_vmd(fixed_path)
    if m is None:
        raise RuntimeError("VMD dosyası okunamadı. Dosya biçimi desteklenmiyor veya bozuk.")

    # İstenirse derinlik hizasını kaldır
    if remove_depth:
        remove_depth_alignment(m, smooth_window=depth_smooth_window, scale=depth_scale)
    # İstenirse ground stabilization uygula
    if stabilize_ground_flag:
        stabilize_ground(m, target_y=ground_target_y, use_feet_only=ground_use_feet_only,
                         smooth_window=ground_smooth_window, scale=ground_scale)

    # Kemik motionları
    bone_channels: Dict[str, List[BoneKey]] = defaultdict(list)
    for f in m.bones:
        bone_channels[f.name].append(
            BoneKey(
                frame=int(f.frame),
                loc=(float(f.pos[0]), float(f.pos[1]), float(f.pos[2])),
                rot=(float(f.quat[0]), float(f.quat[1]), float(f.quat[2]), float(f.quat[3]))
            )
        )

    # Morph motionları
    morph_channels: Dict[str, List[MorphKey]] = defaultdict(list)
    for f in m.morphs:
        morph_channels[f.name].append(MorphKey(frame=int(f.frame), weight=float(f.weight)))

    # Kemik kanallarını optimize et
    new_bone_frames: List[BoneFrame] = []
    rot_eps_rad = math.radians(rot_eps_deg)

    bone_items = list(bone_channels.items())
    if progress is None:
        iterator = tqdm(bone_items, desc='Bones')
    else:
        iterator = bone_items
    for idx_bone, (bone, keys) in enumerate(iterator, start=1):
        keys.sort(key=lambda k: k.frame)

        # opsiyonel kaba downsample
        if key_step > 1 and len(keys) > 2:
            keys = [k for i, k in enumerate(keys) if i == 0 or i == len(keys)-1 or (keys[i].frame - keys[0].frame) % key_step == 0]

        pos_keys = [(k.frame, np.array(k.loc, dtype=np.float32)) for k in keys]
        rot_keys = [(k.frame, np.array(k.rot, dtype=np.float32)) for k in keys]

        simp_pos = simplify_curve(pos_keys, pos_eps)
        simp_rot = simplify_quat_curve(rot_keys, rot_eps_rad)

        # uçları koru
        if preserve_end_keys:
            first_f = keys[0].frame
            last_f = keys[-1].frame
            if simp_pos[0][0] != first_f:
                simp_pos = [(first_f, pos_keys[0][1])] + simp_pos
            if simp_pos[-1][0] != last_f:
                simp_pos = simp_pos + [(last_f, pos_keys[-1][1])]
            if simp_rot[0][0] != first_f:
                simp_rot = [(first_f, rot_keys[0][1])] + simp_rot
            if simp_rot[-1][0] != last_f:
                simp_rot = simp_rot + [(last_f, rot_keys[-1][1])]

        # kare -> değer sözlüğü birleştir
        pos_map = {f: v for f, v in simp_pos}
        rot_map = {f: v for f, v in simp_rot}
        merged_frames = sorted(set(pos_map.keys()) | set(rot_map.keys()))

        for fr in merged_frames:
            p = pos_map.get(fr)
            if p is None:
                # lineer interpolasyon
                prev = max([f for f, _ in simp_pos if f <= fr])
                nxt = min([f for f, _ in simp_pos if f >= fr])
                if prev == nxt:
                    p = pos_map[prev]
                else:
                    t = (fr - prev) / float(nxt - prev)
                    p = pos_map[prev] * (1 - t) + pos_map[nxt] * t
            r = rot_map.get(fr)
            if r is None:
                prev = max([f for f, _ in simp_rot if f <= fr])
                nxt = min([f for f, _ in simp_rot if f >= fr])
                if prev == nxt:
                    r = rot_map[prev]
                else:
                    t = (fr - prev) / float(nxt - prev)
                    r = slerp(rot_map[prev], rot_map[nxt], t)

            new_bone_frames.append(
                BoneFrame(
                    name=bone,
                    frame=int(fr),
                    pos=np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32),
                    quat=np.array([float(r[0]), float(r[1]), float(r[2]), float(r[3])], dtype=np.float32),
                    interp=b"\x00" * 64,
                )
            )
        if progress is not None:
            progress('Bones', idx_bone, len(bone_items))

    # Morph kanallarını optimize et
    new_morph_frames: List[MorphFrame] = []
    morph_items = list(morph_channels.items())
    if progress is None:
        m_iterator = tqdm(morph_items, desc='Morphs')
    else:
        m_iterator = morph_items
    for idx_m, (morph, keys) in enumerate(m_iterator, start=1):
        keys.sort(key=lambda k: k.frame)

        # ufak değerleri sıfırla ve gereksiz anahtarları at
        cleaned = []
        last_w = None
        for k in keys:
            w = 0.0 if abs(k.weight) < morph_eps else k.weight
            if last_w is None or abs(w - last_w) > morph_eps:
                cleaned.append((k.frame, np.array([w], dtype=np.float32)))
                last_w = w
        if len(cleaned) <= 1:
            if cleaned:
                new_morph_frames.append(MorphFrame(name=morph, frame=int(cleaned[0][0]), weight=float(cleaned[0][1][0])))
            continue

        simp = simplify_curve(cleaned, morph_eps)

        if preserve_end_keys:
            first_f = keys[0].frame
            last_f = keys[-1].frame
            if simp[0][0] != first_f:
                simp = [(first_f, cleaned[0][1])] + simp
            if simp[-1][0] != last_f:
                simp = simp + [(last_f, cleaned[-1][1])]

        for fr, val in simp:
            new_morph_frames.append(MorphFrame(name=morph, frame=int(fr), weight=float(val[0])))
        if progress is not None:
            progress('Morphs', idx_m, len(morph_items))

    out_model_name = m.model_name
    if replace_xr_with and (out_model_name.strip() == 'XR Animator'):
        out_model_name = replace_xr_with

    new_motion = Motion(
        model_name=out_model_name,
        bones=sorted(new_bone_frames, key=lambda b: (b.name, b.frame)),
        morphs=sorted(new_morph_frames, key=lambda b: (b.name, b.frame)),
    )

    write_vmd(output_path, new_motion)

    return output_path


def main():
    ap = argparse.ArgumentParser(description='VMD motion optimizasyonu (RDP/SLERP). VMD Motion Optimizer by Barış Keser (barkeser2002)')
    ap.add_argument('input', help='.vmd dosya yolu')
    ap.add_argument('-o', '--output', default=None, help='çıktı .vmd dosyası (varsayılan: <input>_optimized.vmd)')
    ap.add_argument('--pos-eps', type=float, default=0.05, help='pozisyon toleransı')
    ap.add_argument('--rot-eps-deg', type=float, default=0.5, help='rotasyon toleransı (derece)')
    ap.add_argument('--morph-eps', type=float, default=1e-3, help='morph toleransı')
    ap.add_argument('--key-step', type=int, default=1, help='kaba downsample adımı (örn. 2=her 2 frame)')
    ap.add_argument('--no-preserve-end', action='store_true', help='kanal uçlarını koruma')
    # Depth options
    ap.add_argument('--remove-depth', action='store_true', help='global Z hizasını kaldır')
    ap.add_argument('--depth-smooth', type=int, default=0, help='depth için smooth window')
    ap.add_argument('--depth-scale', type=float, default=1.0, help='depth ölçek')
    # Ground options
    ap.add_argument('--stabilize-ground', action='store_true', help='zemine sabitle')
    ap.add_argument('--ground-target-y', type=float, default=0.0, help='hedef zemin Y')
    ap.add_argument('--ground-smooth', type=int, default=0, help='zemin için smooth window')
    ap.add_argument('--ground-scale', type=float, default=1.0, help='zemin ofset ölçek')
    ap.add_argument('--ground-all-bones', action='store_true', help='tüm kemikleri kullan (varsayılan: sadece ayak)')
    # Model adı düzeltme
    ap.add_argument('--replace-xr-with', type=str, default='Barış Keser', help='"XR Animator" model adını bununla değiştir')
    args = ap.parse_args()

    output = args.output or args.input.replace('.vmd', '_optimized.vmd')

    optimize_vmd(
        input_path=args.input,
        output_path=output,
        pos_eps=args.pos_eps,
        rot_eps_deg=args.rot_eps_deg,
        morph_eps=args.morph_eps,
        key_step=args.key_step,
        preserve_end_keys=not args.no_preserve_end,
        remove_depth=args.remove_depth,
        depth_smooth_window=args.depth_smooth,
        depth_scale=args.depth_scale,
        stabilize_ground_flag=args.stabilize_ground,
        ground_target_y=args.ground_target_y,
        ground_use_feet_only=not args.ground_all_bones,
        ground_smooth_window=args.ground_smooth,
        ground_scale=args.ground_scale,
        replace_xr_with=args.replace_xr_with,
    )

    print('Kaydedildi:', output)


if __name__ == '__main__':
    main()

import csv
import json
import os
import platform
import shutil
import sqlite3
import urllib.request
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
try:
    import requests
except Exception:
    requests = None
from PyQt5.QtCore import QDateTime, QPoint, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QRadioButton,
    QScrollBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import serial
except Exception:
    serial = None


APP_NAME = "边缘AI质检工位"
DEVICE_NAME = "边缘智能平台 v2.0"
DEFAULT_MODEL = "best.pt"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CONF_THRES = 0.30
DEFAULT_RECHECK_CONF = 0.55
DEFAULT_RULE_MAX_ALLOWED_MM = 0.10
DEFAULT_RULE_PASS_MM = 0.05
DEFAULT_MM_PER_PIXEL = 0.01
LLM_API_URL = "http://localhost:11434/api/generate"
EDGE_RUNTIME_PROFILE = "edge_arm"
DESKTOP_RUNTIME_PROFILE = "desktop"
SUPPORTED_MODEL_SUFFIXES = {".pt", ".onnx"}
DEPLOY_ONLY_SUFFIXES = {".rknn"}

DEFAULT_LLM_MODELS = {
    "qwen2.5_0.5b": {
        "provider": "ollama",
        "model_name": "qwen2.5:0.5b",
        "label": "Qwen2.5 0.5B",
        "enabled": True,
    },
    "qwen2.5_1.5b": {
        "provider": "ollama",
        "model_name": "qwen2.5:1.5b",
        "label": "Qwen2.5 1.5B",
        "enabled": True,
    },
    "qwen2.5_7b": {
        "provider": "ollama",
        "model_name": "qwen2.5:7b",
        "label": "Qwen2.5 7B",
        "enabled": True,
    },
    "phi3_mini": {
        "provider": "ollama",
        "model_name": "phi3:mini",
        "label": "Phi-3 Mini",
        "enabled": True,
    },
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "edge_runtime"
CAPTURE_DIR = DATA_DIR / "captures"
REPORT_DIR = DATA_DIR / "reports"
EXPORT_DIR = DATA_DIR / "exports"
CONFIG_PATH = DATA_DIR / "station_config.json"
CSV_PATH = REPORT_DIR / "inspection_records.csv"
JSONL_PATH = REPORT_DIR / "inspection_records.jsonl"
SQLITE_PATH = REPORT_DIR / "inspection_records.db"
RKNN_TARGET_PLATFORMS = [
    "rk3588",
    "rk3576",
    "rk3566",
    "rk3568",
    "rk3562",
    "rv1103",
    "rv1106",
    "rv1103b",
    "rv1106b",
]


def default_quality_rule() -> Dict[str, float]:
    return {
        "max_allowed_mm": DEFAULT_RULE_MAX_ALLOWED_MM,
        "pass_mm": DEFAULT_RULE_PASS_MM,
        "zero_tolerance": False,
    }


def extract_model_class_names(model) -> List[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def detect_runtime_profile() -> str:
    machine = platform.machine().lower()
    if any(token in machine for token in ("arm", "aarch64")):
        return EDGE_RUNTIME_PROFILE
    return DESKTOP_RUNTIME_PROFILE


def ensure_runtime_dirs() -> None:
    for path in (DATA_DIR, CAPTURE_DIR, REPORT_DIR, EXPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_csv_header_if_needed() -> None:
    header = [
        "record_id",
        "timestamp",
        "station",
        "operator",
        "source_type",
        "source_name",
        "detector_status",
        "decision",
        "action",
        "defect_count",
        "top_confidence",
        "avg_confidence",
        "classes",
        "inference_ms",
        "latency_ms",
        "image_path",
        "notes",
        "analysis",
    ]
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
            csv.writer(file).writerow(header)
        return

    try:
        with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.reader(file))
    except Exception:
        return

    if not rows:
        with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
            csv.writer(file).writerow(header)
        return

    if rows[0] == header:
        return

    if rows[0] and rows[0][0] == "timestamp":
        migrated_rows = [header]
        for row in rows[1:]:
            migrated_rows.append([""] + row)
        with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
            csv.writer(file).writerows(migrated_rows)


def ensure_sqlite_db() -> None:
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspection_records (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                station TEXT NOT NULL,
                operator TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                detector_status TEXT NOT NULL,
                decision TEXT NOT NULL,
                action TEXT NOT NULL,
                defect_count INTEGER NOT NULL,
                top_confidence REAL NOT NULL,
                avg_confidence REAL NOT NULL,
                classes_json TEXT NOT NULL,
                inference_ms REAL NOT NULL,
                latency_ms REAL NOT NULL,
                image_path TEXT NOT NULL,
                notes TEXT NOT NULL,
                analysis TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_inspection_records_timestamp
            ON inspection_records(timestamp)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS human_reviews (
                review_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL,
                human_label TEXT NOT NULL,
                review_note TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT NOT NULL,
                FOREIGN KEY(record_id) REFERENCES inspection_records(record_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_human_reviews_record_id
            ON human_reviews(record_id)
            """
        )
        conn.commit()


@dataclass
class RuleSpec:
    max_allowed_mm: float
    pass_mm: float
    zero_tolerance: bool = False


@dataclass
class StationConfig:
    station_name: str = "产线工位 A-01"
    operator: str = "边缘AI平台"
    default_model: str = str((BASE_DIR / DEFAULT_MODEL).resolve())
    runtime_profile: str = EDGE_RUNTIME_PROFILE
    camera_index: int = DEFAULT_CAMERA_INDEX
    confidence: float = DEFAULT_CONF_THRES
    recheck_confidence: float = DEFAULT_RECHECK_CONF
    input_width: int = 640
    input_height: int = 640
    frame_interval_ms: int = 150
    auto_fullscreen: bool = False
    enable_ai_analysis: bool = False
    auto_analyze_on_alert: bool = False
    llm_api_url: str = LLM_API_URL
    llm_models: Dict[str, Dict[str, object]] = field(default_factory=lambda: dict(DEFAULT_LLM_MODELS))
    active_llm_model: str = "qwen2.5_0.5b"
    enable_actuator: bool = False
    actuator_mode: str = "simulate"
    serial_port: str = ""
    serial_baudrate: int = 9600
    stop_line_ng_threshold: int = 10
    confirm_frames: int = 2
    calibration_reference_px: int = 100
    calibration_reference_mm: float = 1.0
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL
    quality_rules: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "scratch": default_quality_rule(),
            "glue_overflow": default_quality_rule(),
            "pinhole": {"max_allowed_mm": 0.0, "pass_mm": 0.0, "zero_tolerance": True},
        }
    )

    @classmethod
    def load(cls, path: Path) -> "StationConfig":
        default = cls(runtime_profile=detect_runtime_profile())
        if not path.exists():
            path.write_text(json.dumps(asdict(default), ensure_ascii=False, indent=2), encoding="utf-8")
            return default
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            merged = asdict(default)
            merged.update(raw)
            if "llm_models" not in merged or not merged["llm_models"]:
                merged["llm_models"] = dict(DEFAULT_LLM_MODELS)
            calibration_px = int(merged.get("calibration_reference_px", 0) or 0)
            calibration_mm = float(merged.get("calibration_reference_mm", 0.0) or 0.0)
            if calibration_px > 0 and calibration_mm > 0:
                merged["mm_per_pixel"] = calibration_mm / calibration_px
            return cls(**merged)
        except Exception:
            return default

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def resolved_mm_per_pixel(self) -> float:
        if self.calibration_reference_px > 0 and self.calibration_reference_mm > 0:
            return float(self.calibration_reference_mm) / float(self.calibration_reference_px)
        return max(float(self.mm_per_pixel), 0.000001)

    def compiled_rules(self) -> Dict[str, RuleSpec]:
        rules: Dict[str, RuleSpec] = {}
        for defect_type, value in self.quality_rules.items():
            rules[defect_type] = RuleSpec(
                max_allowed_mm=float(value.get("max_allowed_mm", 0.0)),
                pass_mm=float(value.get("pass_mm", 0.0)),
                zero_tolerance=bool(value.get("zero_tolerance", False)),
            )
        return rules

    def ensure_quality_rules(self, defect_types: List[str]) -> List[str]:
        added: List[str] = []
        for defect_type in defect_types:
            normalized = str(defect_type).strip()
            if not normalized or normalized in self.quality_rules:
                continue
            self.quality_rules[normalized] = default_quality_rule()
            added.append(normalized)
        return added

    def resolve_active_llm(self) -> Dict[str, object]:
        model_entry = self.llm_models.get(self.active_llm_model, {})
        return {
            "key": self.active_llm_model,
            "provider": str(model_entry.get("provider", "ollama")),
            "model_name": str(model_entry.get("model_name", "")),
            "label": str(model_entry.get("label", self.active_llm_model)),
            "enabled": bool(model_entry.get("enabled", False)),
        }


@dataclass
class DetectionBox:
    defect_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    estimated_size_mm: float
    trigger_recheck: bool = False


@dataclass
class InspectionDecision:
    detector_status: str
    decision: str
    action: str
    notes: str


@dataclass
class EdgeAgentState:
    total_count: int = 0
    ok_count: int = 0
    recheck_count: int = 0
    ng_count: int = 0


@dataclass
class InspectionRecord:
    record_id: str
    timestamp: str
    station: str
    operator: str
    source_type: str
    source_name: str
    detector_status: str
    decision: str
    action: str
    defect_count: int
    top_confidence: float
    avg_confidence: float
    classes: List[str]
    inference_ms: float
    latency_ms: float
    image_path: str
    notes: str
    analysis: str = ""

    def to_row(self) -> List[str]:
        return [
            self.record_id,
            self.timestamp,
            self.station,
            self.operator,
            self.source_type,
            self.source_name,
            self.detector_status,
            self.decision,
            self.action,
            self.defect_count,
            f"{self.top_confidence:.4f}",
            f"{self.avg_confidence:.4f}",
            ",".join(self.classes),
            f"{self.inference_ms:.2f}",
            f"{self.latency_ms:.2f}",
            self.image_path,
            self.notes,
            self.analysis,
        ]

    def to_json(self) -> Dict[str, object]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "station": self.station,
            "operator": self.operator,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "detector_status": self.detector_status,
            "decision": self.decision,
            "action": self.action,
            "defect_count": self.defect_count,
            "top_confidence": round(self.top_confidence, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "classes": self.classes,
            "inference_ms": round(self.inference_ms, 2),
            "latency_ms": round(self.latency_ms, 2),
            "image_path": self.image_path,
            "notes": self.notes,
            "analysis": self.analysis,
        }

    def to_db_row(self) -> Tuple[object, ...]:
        return (
            self.record_id,
            self.timestamp,
            self.station,
            self.operator,
            self.source_type,
            self.source_name,
            self.detector_status,
            self.decision,
            self.action,
            self.defect_count,
            round(self.top_confidence, 4),
            round(self.avg_confidence, 4),
            json.dumps(self.classes, ensure_ascii=False),
            round(self.inference_ms, 2),
            round(self.latency_ms, 2),
            self.image_path,
            self.notes,
            self.analysis,
        )


class DecisionEngine:
    def __init__(self, rules: Dict[str, RuleSpec], recheck_confidence: float) -> None:
        self.rules = rules
        self.recheck_confidence = recheck_confidence

    def decide(self, detections: List[DetectionBox]) -> InspectionDecision:
        if not detections:
            return InspectionDecision("OK", "OK", "PASS", "未检测到缺陷")

        unknown_types = [box.defect_type for box in detections if box.defect_type not in self.rules]
        if unknown_types:
            return InspectionDecision("NG", "NG", "HOLD", "存在未配置规则的缺陷类型，按风险优先判定为 NG")

        for box in detections:
            rule = self.rules[box.defect_type]
            if rule.zero_tolerance:
                return InspectionDecision("NG", "NG", "REJECT", f"{box.defect_type} 为零容忍缺陷")
            if box.estimated_size_mm > rule.max_allowed_mm:
                return InspectionDecision("NG", "NG", "REJECT", f"{box.defect_type} 超出允许阈值")

        recheck_hits = [box for box in detections if box.trigger_recheck or box.confidence < self.recheck_confidence]
        if recheck_hits:
            return InspectionDecision("NG", "RECHECK", "RECHECK", "存在低置信度或灰区缺陷，建议人工复检")

        for box in detections:
            rule = self.rules[box.defect_type]
            if box.estimated_size_mm > rule.pass_mm:
                return InspectionDecision("NG", "RECHECK", "RECHECK", f"{box.defect_type} 接近上限，进入复检")

        return InspectionDecision("NG", "NG", "REJECT", "检测到缺陷且不满足放行条件")


class ActuatorController:
    def __init__(self, config: StationConfig) -> None:
        self.config = config
        self.consecutive_ng = 0
        self.serial_client = None

    def _get_serial_client(self):
        if not self.config.enable_actuator or self.config.actuator_mode != "serial":
            return None
        if serial is None or not self.config.serial_port:
            return None
        if self.serial_client is None:
            self.serial_client = serial.Serial(self.config.serial_port, self.config.serial_baudrate, timeout=1)
        return self.serial_client

    def execute(self, decision: InspectionDecision) -> Tuple[str, str]:
        action = decision.action
        if action in {"PASS", "RECHECK"}:
            self.consecutive_ng = 0
        elif action in {"REJECT", "HOLD"}:
            self.consecutive_ng += 1

        if self.consecutive_ng >= self.config.stop_line_ng_threshold:
            action = "STOP_LINE"

        if not self.config.enable_actuator:
            return action, f"执行器未启用，当前以记录模式输出动作: {action}"
        if self.config.actuator_mode == "simulate":
            return action, f"模拟执行动作: {action}"
        if self.config.actuator_mode == "serial":
            try:
                client = self._get_serial_client()
                if client is None:
                    return action, "串口执行器不可用，已降级为记录模式"
                client.write(f"{action}\n".encode("utf-8"))
                return action, f"串口动作已发送: {action}"
            except Exception as exc:
                return action, f"串口动作发送失败: {exc}"
        return action, f"未知执行模式: {self.config.actuator_mode}"

    def close(self) -> None:
        if self.serial_client is not None:
            try:
                self.serial_client.close()
            except Exception:
                pass
            self.serial_client = None


class EdgeAgent:
    def __init__(self, decision_engine: DecisionEngine, actuator: ActuatorController, config: StationConfig) -> None:
        self.decision_engine = decision_engine
        self.actuator = actuator
        self.config = config
        self.state = EdgeAgentState()

    def process_detection(self, payload: dict) -> Tuple[InspectionDecision, str]:
        decision = self.decision_engine.decide(payload["boxes"])
        action, detail = self.actuator.execute(decision)
        decision.action = action

        self.state.total_count += 1
        if decision.decision == "OK":
            self.state.ok_count += 1
        elif decision.decision == "RECHECK":
            self.state.recheck_count += 1
        else:
            self.state.ng_count += 1
        return decision, detail

    def pass_rate(self) -> float:
        return self.state.ok_count / self.state.total_count * 100 if self.state.total_count else 0.0

    def active_llm_label(self) -> str:
        return str(self.config.resolve_active_llm().get("label", self.config.active_llm_model))

    def active_llm_model_name(self) -> str:
        return str(self.config.resolve_active_llm().get("model_name", ""))

    def llm_enabled(self) -> bool:
        return self.config.enable_ai_analysis and bool(self.config.resolve_active_llm().get("enabled", False))

    def ai_unavailable_reason(self) -> str:
        if not self.config.enable_ai_analysis:
            return "当前运行档位已关闭本地 AI 辅助分析。"
        model_entry = self.config.resolve_active_llm()
        if not model_entry.get("enabled", False):
            return f"当前选定模型未启用: {model_entry.get('label', self.config.active_llm_model)}"
        if not model_entry.get("model_name"):
            return "当前未配置有效的本地模型名。"
        return ""

    def should_trigger_ai(self, decision: InspectionDecision) -> bool:
        return decision.decision != "OK" and self.config.auto_analyze_on_alert and self.llm_enabled()


class IntegratedAgentService:
    def __init__(self, db_path: Path, config: StationConfig, edge_agent: EdgeAgent) -> None:
        self.db_path = db_path
        self.config = config
        self.edge_agent = edge_agent

    def fetch_recent_similar_records(self, classes: List[str], limit: int = 5) -> List[Dict[str, object]]:
        if not classes:
            return []
        records: List[Dict[str, object]] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, decision, defect_count, top_confidence, image_path, notes, classes_json
                FROM inspection_records
                ORDER BY timestamp DESC
                LIMIT 50
                """
            )
            for row in cursor.fetchall():
                row_classes = json.loads(row[6]) if row[6] else []
                if not set(classes).intersection(row_classes):
                    continue
                records.append(
                    {
                        "timestamp": row[0],
                        "decision": row[1],
                        "defect_count": row[2],
                        "top_confidence": row[3],
                        "image_path": row[4],
                        "notes": row[5],
                        "classes": row_classes,
                    }
                )
                if len(records) >= limit:
                    break
        return records

    def build_shift_summary(self, limit: int = 200) -> str:
        with sqlite3.connect(self.db_path) as conn:
            record_rows = conn.execute(
                """
                SELECT timestamp, decision, defect_count, top_confidence, classes_json
                FROM inspection_records
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            review_rows = conn.execute(
                """
                SELECT human_label, COUNT(*)
                FROM human_reviews
                GROUP BY human_label
                ORDER BY COUNT(*) DESC
                """
            ).fetchall()

        if not record_rows:
            return "班次总结\n暂无检测记录。"

        total = len(record_rows)
        decision_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {}
        confidence_sum = 0.0
        defect_total = 0
        for row in record_rows:
            decision = str(row[1])
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            defect_total += int(row[2])
            confidence_sum += float(row[3])
            classes = json.loads(row[4]) if row[4] else []
            for defect_type in classes:
                class_counts[str(defect_type)] = class_counts.get(str(defect_type), 0) + 1

        top_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        avg_confidence = confidence_sum / total if total else 0.0
        review_text = "无"
        if review_rows:
            review_text = " / ".join(f"{label}:{count}" for label, count in review_rows)

        lines = [
            "班次总结",
            f"统计样本: 最近 {total} 条检测记录",
            f"判定分布: " + " / ".join(f"{key}:{value}" for key, value in sorted(decision_counts.items())),
            f"缺陷总数: {defect_total}",
            f"平均最高置信度: {avg_confidence:.2f}",
            "缺陷Top5: " + (" / ".join(f"{name}:{count}" for name, count in top_classes) if top_classes else "无"),
            f"人工复判分布: {review_text}",
        ]
        return "\n".join(lines)

    def build_rule_suggestions(self, limit: int = 300) -> str:
        with sqlite3.connect(self.db_path) as conn:
            review_rows = conn.execute(
                """
                SELECT hr.human_label, ir.classes_json, ir.top_confidence, ir.decision
                FROM human_reviews hr
                JOIN inspection_records ir ON hr.record_id = ir.record_id
                ORDER BY hr.reviewed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        if not review_rows:
            return "规则建议\n暂无人工复判数据，无法生成建议。"

        class_stats: Dict[str, Dict[str, object]] = {}
        for human_label, classes_json, top_confidence, decision in review_rows:
            classes = json.loads(classes_json) if classes_json else []
            for defect_type in classes:
                stats = class_stats.setdefault(
                    str(defect_type),
                    {
                        "total": 0,
                        "false_positive": 0,
                        "confirmed_ng": 0,
                        "manual_recheck": 0,
                        "confidence_sum": 0.0,
                    },
                )
                stats["total"] = int(stats["total"]) + 1
                stats["confidence_sum"] = float(stats["confidence_sum"]) + float(top_confidence)
                if human_label == "误报":
                    stats["false_positive"] = int(stats["false_positive"]) + 1
                elif human_label == "NG确认":
                    stats["confirmed_ng"] = int(stats["confirmed_ng"]) + 1
                elif human_label == "人工复检":
                    stats["manual_recheck"] = int(stats["manual_recheck"]) + 1

        lines = ["规则建议"]
        for defect_type, stats in sorted(class_stats.items(), key=lambda item: int(item[1]["total"]), reverse=True):
            total = int(stats["total"])
            false_positive = int(stats["false_positive"])
            confirmed_ng = int(stats["confirmed_ng"])
            manual_recheck = int(stats["manual_recheck"])
            avg_conf = float(stats["confidence_sum"]) / total if total else 0.0

            suggestion = "保持当前规则"
            if false_positive / total >= 0.4:
                suggestion = "误报偏多，建议适当提高置信度阈值或收紧尺寸放行规则"
            elif confirmed_ng / total >= 0.7 and avg_conf >= 0.7:
                suggestion = "异常较稳定，建议评估是否收紧 pass_mm / max_allowed_mm"
            elif manual_recheck / total >= 0.4:
                suggestion = "人工复检比例偏高，建议细化该类规则分段或增加连续帧确认"

            lines.append(
                f"{defect_type}: 复判{total}次 | 误报={false_positive} | NG确认={confirmed_ng} | "
                f"人工复检={manual_recheck} | 平均置信度={avg_conf:.2f} | 建议={suggestion}"
            )

        return "\n".join(lines)

    def build_analysis_prompt(
        self,
        payload: Optional[dict],
        station_name: str,
        station_operator: str,
    ) -> str:
        summary = "暂无检测结果"
        history_block = "最近同类异常: 暂无"
        if payload:
            classes = payload.get("classes", [])
            summary = (
                f"检测状态: {payload.get('detector_status', 'UNKNOWN')}\n"
                f"缺陷数量: {payload.get('defect_count', 0)}\n"
                f"缺陷类别: {'、'.join(classes) if classes else '暂无'}\n"
                f"最高置信度: {payload.get('top_confidence', 0.0):.2f}\n"
                f"估算最大尺寸: {payload.get('max_estimated_size_mm', 0.0):.3f} mm\n"
                f"当前标定: {payload.get('mm_per_pixel', self.config.resolved_mm_per_pixel()):.6f} mm/px\n"
                f"推理耗时: {payload.get('inference_ms', 0.0):.1f} ms\n"
            )
            recent_records = self.fetch_recent_similar_records(classes)
            if recent_records:
                history_lines = []
                for index, record in enumerate(recent_records, start=1):
                    history_lines.append(
                        f"{index}. {record['timestamp']} | {record['decision']} | 缺陷={record['defect_count']} | "
                        f"置信度={record['top_confidence']:.2f} | 类别={'、'.join(record['classes'])}"
                    )
                history_block = "最近同类异常:\n" + "\n".join(history_lines)

        return (
            "你是一名工业质检现场支持工程师。请基于以下本地检测结果，给出保守、可执行的排查建议，不要把推测写成确定结论。\n\n"
            f"设备: {DEVICE_NAME}\n"
            f"工位: {station_name}\n"
            f"操作员: {station_operator}\n"
            f"辅助模型: {self.edge_agent.active_llm_label()}\n"
            f"当前规则标定: {self.config.resolved_mm_per_pixel():.6f} mm/px\n"
            f"{summary}\n"
            f"{history_block}\n\n"
            "请输出:\n"
            "1. 当前异常说明\n"
            "2. 可能原因（明确写为可能）\n"
            "3. 现场排查顺序\n"
            "4. 是否建议停线或人工复检\n"
        )

    def should_escalate_stop_line(
        self,
        decision: str,
        action: str,
        consecutive_ng: int,
        stop_line_threshold: int,
        classes: List[str],
        limit: int = 10,
    ) -> Dict[str, object]:
        if decision == "OK":
            return {
                "level": "normal",
                "label": "正常监控",
                "message": "停线建议: 当前结果正常，不建议停线。",
            }

        with sqlite3.connect(self.db_path) as conn:
            recent_rows = conn.execute(
                """
                SELECT decision, classes_json
                FROM inspection_records
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        recent_related = 0
        recent_abnormal = 0
        for row in recent_rows:
            row_classes = json.loads(row[1]) if row[1] else []
            if classes and not set(classes).intersection(row_classes):
                continue
            recent_related += 1
            if row[0] in {"NG", "RECHECK"}:
                recent_abnormal += 1

        abnormal_ratio = (recent_abnormal / recent_related) if recent_related else 0.0
        threshold_gap = stop_line_threshold - consecutive_ng

        if action == "STOP_LINE" or consecutive_ng >= stop_line_threshold:
            return {
                "level": "stop",
                "label": "建议停线",
                "message": (
                    f"停线建议: 已达到停线阈值。连续异常 {consecutive_ng}/{stop_line_threshold}，"
                    f"建议立即停线并安排人工复核。"
                ),
            }
        if threshold_gap <= 1:
            return {
                "level": "warning",
                "label": "接近停线",
                "message": (
                    f"停线建议: 已接近停线阈值。连续异常 {consecutive_ng}/{stop_line_threshold}，"
                    f"建议准备停线并优先排查当前缺陷。"
                ),
            }
        if abnormal_ratio >= 0.7 and recent_related >= 3:
            return {
                "level": "warning",
                "label": "接近停线",
                "message": (
                    f"停线建议: 最近同类异常比例偏高 ({recent_abnormal}/{recent_related})，"
                    "建议安排人工复检并评估是否预防性停线。"
                ),
            }
        return {
            "level": "normal",
            "label": "正常监控",
            "message": (
                f"停线建议: 暂未达到停线阈值。连续异常 {consecutive_ng}/{stop_line_threshold}，"
                "建议继续监控并保留人工复检。"
            ),
        }

    def record_human_review(self, record_id: str, human_label: str, review_note: str) -> str:
        reviewed_at = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO human_reviews (review_id, record_id, human_label, review_note, reviewed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uuid.uuid4().hex, record_id, human_label, review_note, reviewed_at),
            )
            conn.commit()
        return (
            f"人工复判已记录\n记录ID: {record_id}\n复判结果: {human_label}\n"
            f"备注: {review_note or '无'}\n时间: {reviewed_at}"
        )

    def run(self, action: str, **kwargs) -> str:
        if action == "analysis_prompt":
            return self.build_analysis_prompt(
                kwargs.get("payload"),
                kwargs.get("station_name", ""),
                kwargs.get("station_operator", ""),
            )
        if action == "shift_summary":
            return self.build_shift_summary(limit=int(kwargs.get("limit", 200)))
        if action == "rule_suggestions":
            return self.build_rule_suggestions(limit=int(kwargs.get("limit", 300)))
        if action == "human_review":
            return self.record_human_review(
                kwargs["record_id"],
                kwargs["human_label"],
                kwargs.get("review_note", ""),
            )
        if action == "stop_line_advice":
            return json.dumps(
                self.should_escalate_stop_line(
                    kwargs.get("decision", "UNKNOWN"),
                    kwargs.get("action", ""),
                    int(kwargs.get("consecutive_ng", 0)),
                    int(kwargs.get("stop_line_threshold", self.config.stop_line_ng_threshold)),
                    kwargs.get("classes", []),
                    limit=int(kwargs.get("limit", 10)),
                ),
                ensure_ascii=False,
            )
        raise ValueError(f"Unsupported integrated agent action: {action}")


class LLMAnalysisWorker(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, prompt: str, api_url: str, model_name: str) -> None:
        super().__init__()
        self.prompt = prompt
        self.api_url = api_url
        self.model_name = model_name

    def run(self) -> None:
        try:
            payload = {"model": self.model_name, "prompt": self.prompt, "stream": False}
            if requests is not None:
                response = requests.post(self.api_url, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
            else:
                request = urllib.request.Request(
                    self.api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
            self.result_signal.emit(data.get("response", "LLM 未返回有效内容。"))
        except Exception as exc:
            self.result_signal.emit(f"AI 辅助分析失败: {exc}")


class DetectorWorker(QThread):
    frame_signal = pyqtSignal(QImage)
    result_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    stopped_signal = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.running = False
        self.model = None
        self.yolo_cls = None
        self.current_model_path = ""
        self.source_type = "camera"
        self.source_path = ""
        self.cap = None
        self.camera_index = DEFAULT_CAMERA_INDEX
        self.conf_thres = DEFAULT_CONF_THRES
        self.recheck_confidence = DEFAULT_RECHECK_CONF
        self.input_width = 640
        self.input_height = 640
        self.frame_interval_ms = 150
        self.runtime_profile = EDGE_RUNTIME_PROFILE
        self.confirm_frames = 2
        self.mm_per_pixel = DEFAULT_MM_PER_PIXEL
        self.detection_streak = 0
        self.last_detection_signature: Tuple[str, ...] = ()

    def load_model(self, weight_path: str) -> bool:
        model_path = Path(weight_path)
        suffix = model_path.suffix.lower()

        if suffix in DEPLOY_ONLY_SUFFIXES:
            self.log_signal.emit("[模型] 当前 GUI 未内置 RKNN 后处理链路，请在平台 SDK 中接入 NPU 推理。")
            return False
        if suffix not in SUPPORTED_MODEL_SUFFIXES:
            self.log_signal.emit(f"[模型] 不支持的模型格式: {suffix}")
            return False

        try:
            if self.runtime_profile == EDGE_RUNTIME_PROFILE and suffix == ".pt":
                self.log_signal.emit("[模型] 边缘设备建议优先使用 ONNX；PT 在 ARM CPU 上通常更慢。")
            if self.yolo_cls is None:
                from ultralytics import YOLO

                self.yolo_cls = YOLO
            self.model = self.yolo_cls(str(model_path))
            self.current_model_path = str(model_path.resolve())
            self.log_signal.emit(f"[模型] 加载成功: {model_path.name}")
            return True
        except Exception as exc:
            self.log_signal.emit(f"[模型] 加载失败: {exc}")
            return False

    def export_model(self, export_format: str, target_path: str, rknn_target: str = "rk3588") -> Tuple[bool, str]:
        if not self.model or not self.current_model_path:
            return False, "请先加载 .pt 模型再导出。"

        source_path = Path(self.current_model_path)
        if source_path.suffix.lower() != ".pt":
            return False, "当前仅支持从 .pt 模型导出 ONNX/RKNN。"

        if not hasattr(self.model, "export"):
            return False, "当前模型对象不支持 export() 接口。"

        export_target = Path(target_path)
        export_target.parent.mkdir(parents=True, exist_ok=True)

        try:
            export_kwargs = {
                "format": export_format,
                "imgsz": [self.input_height, self.input_width],
            }
            if export_format == "onnx":
                export_kwargs["simplify"] = True
            elif export_format == "rknn":
                export_kwargs["name"] = rknn_target.lower().strip() or "rk3588"

            exported = Path(str(self.model.export(**export_kwargs)))
            if not exported.exists():
                return False, f"导出结束，但未找到输出结果: {exported}"

            if export_format == "onnx":
                if export_target.suffix.lower() != ".onnx":
                    export_target = export_target.with_suffix(".onnx")
                if export_target.exists():
                    export_target.unlink()
                shutil.move(str(exported), str(export_target))
                return True, str(export_target)

            if export_format == "rknn":
                final_dir = export_target
                if final_dir.exists():
                    final_dir = final_dir.parent / f"{final_dir.name}_{int(time.time())}"
                shutil.move(str(exported), str(final_dir))
                return True, str(final_dir)

            return False, f"不支持的导出格式: {export_format}"
        except Exception as exc:
            return False, str(exc)

    def set_source(self, source_type: str, path: str = "") -> None:
        self.source_type = source_type
        self.source_path = path

    def set_camera_index(self, index: int) -> None:
        self.camera_index = index

    def set_confidence(self, value: float) -> None:
        self.conf_thres = value

    def set_recheck_confidence(self, value: float) -> None:
        self.recheck_confidence = value

    def set_runtime(
        self,
        profile: str,
        width: int,
        height: int,
        frame_interval_ms: int,
        confirm_frames: int = 1,
        mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
    ) -> None:
        self.runtime_profile = profile
        self.input_width = width
        self.input_height = height
        self.frame_interval_ms = frame_interval_ms
        self.confirm_frames = max(1, int(confirm_frames))
        self.mm_per_pixel = max(float(mm_per_pixel), 0.000001)

    def estimate_size_mm(self, x1: int, y1: int, x2: int, y2: int) -> float:
        pixel_span = max(max(x2 - x1, 1), max(y2 - y1, 1))
        return pixel_span * self.mm_per_pixel

    def run(self) -> None:
        self.running = True
        self.detection_streak = 0
        self.last_detection_signature = ()
        if not self.model:
            self.log_signal.emit("[错误] 未加载检测模型。")
            self.stopped_signal.emit()
            return

        if self.source_type == "image" and os.path.exists(self.source_path):
            image = cv2.imread(self.source_path)
            if image is None:
                self.log_signal.emit("[错误] 图像读取失败。")
            else:
                self.process_frame(image)
            self.running = False
            self.stopped_signal.emit()
            return

        if self.source_type == "camera":
            self.cap = cv2.VideoCapture(self.camera_index)
        elif self.source_type == "video" and os.path.exists(self.source_path):
            self.cap = cv2.VideoCapture(self.source_path)
        else:
            self.log_signal.emit("[错误] 无效输入源。")
            self.running = False
            self.stopped_signal.emit()
            return

        if not self.cap or not self.cap.isOpened():
            self.log_signal.emit("[错误] 摄像头或视频源打开失败。")
            self.running = False
            self.stopped_signal.emit()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_height)

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.process_frame(frame)
            self.msleep(self.frame_interval_ms)

        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        self.stopped_signal.emit()

    def prepare_frame(self, frame):
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        if (w, h) == (self.input_width, self.input_height):
            return frame
        return cv2.resize(frame, (self.input_width, self.input_height))

    def process_frame(self, frame) -> None:
        loop_start = time.perf_counter()
        frame = self.prepare_frame(frame)
        inference_start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            device="cpu",
            verbose=False,
            imgsz=max(self.input_width, self.input_height),
        )
        inference_ms = (time.perf_counter() - inference_start) * 1000
        result = results[0]
        annotated = result.plot()

        boxes: List[DetectionBox] = []
        classes: List[str] = []
        confidences: List[float] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls_list = result.boxes.cls.cpu().numpy().astype(int)
            conf_list = result.boxes.conf.cpu().numpy()
            for bbox, cls_idx, conf in zip(xyxy, cls_list, conf_list):
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                defect_type = self.model.names[int(cls_idx)]
                classes.append(defect_type)
                confidences.append(float(conf))
                estimated_size_mm = self.estimate_size_mm(x1, y1, x2, y2)
                boxes.append(
                    DetectionBox(
                        defect_type=defect_type,
                        confidence=float(conf),
                        bbox=(x1, y1, x2, y2),
                        estimated_size_mm=estimated_size_mm,
                        trigger_recheck=float(conf) < self.recheck_confidence,
                    )
                )

        classes_signature = tuple(sorted(classes))
        raw_defect_count = len(boxes)
        confirmed = True
        pending_frames = 0
        if raw_defect_count:
            if classes_signature == self.last_detection_signature:
                self.detection_streak += 1
            else:
                self.last_detection_signature = classes_signature
                self.detection_streak = 1
            confirmed = self.detection_streak >= self.confirm_frames
            pending_frames = max(self.confirm_frames - self.detection_streak, 0)
        else:
            self.detection_streak = 0
            self.last_detection_signature = ()

        detector_status = "NG" if raw_defect_count else "OK"
        if raw_defect_count and not confirmed:
            detector_status = "PENDING"
        top_confidence = max(confidences) if confidences else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        max_estimated_size_mm = max((box.estimated_size_mm for box in boxes), default=0.0)
        latency_ms = (time.perf_counter() - loop_start) * 1000

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.frame_signal.emit(qimage)
        self.result_signal.emit(
            {
                "detector_status": detector_status,
                "boxes": boxes if confirmed else [],
                "defect_count": len(boxes) if confirmed else 0,
                "raw_boxes": boxes,
                "raw_defect_count": raw_defect_count,
                "confirmed": confirmed,
                "pending_frames": pending_frames,
                "top_confidence": top_confidence,
                "avg_confidence": avg_confidence,
                "classes": classes,
                "max_estimated_size_mm": max_estimated_size_mm,
                "mm_per_pixel": self.mm_per_pixel,
                "annotated_frame": annotated.copy(),
                "inference_ms": inference_ms,
                "latency_ms": latency_ms,
            }
        )
        self.log_signal.emit(
            f"[检测] 状态={detector_status} 缺陷数={raw_defect_count} 最高置信度={top_confidence:.2f} 推理={inference_ms:.1f}ms"
        )

    def stop(self) -> None:
        self.running = False
        self.msleep(50)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.quit()
        self.wait()


class DataCard(QFrame):
    def __init__(self, title: str, value: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("dataCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardKicker")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("cardValue")
        self.detail_label = QLabel("实时更新")
        self.detail_label.setObjectName("cardDetail")

        accent_bar = QFrame()
        accent_bar.setFixedHeight(4)
        accent_bar.setStyleSheet(f"background:{accent}; border:none; border-radius:2px;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(accent_bar)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)

    def set_detail(self, detail: str) -> None:
        self.detail_label.setText(detail)


class CalibrationPreviewLabel(QLabel):
    point_selected = pyqtSignal(int, int)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._source_pixmap: Optional[QPixmap] = None
        self._source_size: Optional[Tuple[int, int]] = None
        self._calibration_enabled = False
        self._calibration_points: List[Tuple[int, int]] = []

    def set_calibration_points(self, points: List[Tuple[int, int]]) -> None:
        self._calibration_points = list(points)
        self._refresh_scaled_pixmap()

    def set_frame_image(self, image: QImage) -> None:
        self._source_pixmap = QPixmap.fromImage(image)
        self._source_size = (image.width(), image.height())
        self._refresh_scaled_pixmap()

    def set_calibration_enabled(self, enabled: bool) -> None:
        self._calibration_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def mousePressEvent(self, event) -> None:
        if not self._calibration_enabled or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        mapped = self._map_widget_pos_to_image(event.pos())
        if mapped is None:
            return
        self.point_selected.emit(mapped[0], mapped[1])

    def _refresh_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        scaled = self._source_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if self._calibration_points:
            scaled = self._draw_calibration_overlay(scaled)
        super().setPixmap(scaled)

    def _draw_calibration_overlay(self, pixmap: QPixmap) -> QPixmap:
        if self._source_size is None:
            return pixmap
        canvas = QPixmap(pixmap)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)

        scale_x = pixmap.width() / self._source_size[0]
        scale_y = pixmap.height() / self._source_size[1]
        pen = QPen(QColor("#7ae7ff"))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QColor("#f59e0b"))

        mapped_points: List[QPoint] = []
        for point in self._calibration_points:
            mapped = QPoint(int(point[0] * scale_x), int(point[1] * scale_y))
            mapped_points.append(mapped)
            painter.drawEllipse(mapped, 6, 6)

        if len(mapped_points) >= 2:
            painter.drawLine(mapped_points[0], mapped_points[1])
            dx = self._calibration_points[0][0] - self._calibration_points[1][0]
            dy = self._calibration_points[0][1] - self._calibration_points[1][1]
            distance_px = (dx * dx + dy * dy) ** 0.5
            midpoint = QPoint(
                int((mapped_points[0].x() + mapped_points[1].x()) / 2),
                int((mapped_points[0].y() + mapped_points[1].y()) / 2),
            )
            label = f"{distance_px:.1f}px"
            text_rect = painter.fontMetrics().boundingRect(label).adjusted(-8, -4, 8, 4)
            text_rect.moveCenter(midpoint)
            painter.fillRect(text_rect, QColor(6, 17, 27, 220))
            painter.drawText(text_rect, Qt.AlignCenter, label)

        painter.end()
        return canvas

    def _map_widget_pos_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        if self._source_pixmap is None or self._source_size is None:
            return None
        displayed = self.pixmap()
        if displayed is None:
            return None
        x_offset = (self.width() - displayed.width()) / 2
        y_offset = (self.height() - displayed.height()) / 2
        if (
            pos.x() < x_offset
            or pos.y() < y_offset
            or pos.x() > x_offset + displayed.width()
            or pos.y() > y_offset + displayed.height()
        ):
            return None
        image_x = int((pos.x() - x_offset) * self._source_size[0] / displayed.width())
        image_y = int((pos.y() - y_offset) * self._source_size[1] / displayed.height())
        image_x = max(0, min(self._source_size[0] - 1, image_x))
        image_y = max(0, min(self._source_size[1] - 1, image_y))
        return image_x, image_y


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_dirs()
        write_csv_header_if_needed()
        ensure_sqlite_db()

        self.config = StationConfig.load(CONFIG_PATH)
        self.station_name = self.config.station_name
        self.station_operator = self.config.operator
        self.default_model_path = self.config.default_model

        self.decision_engine = DecisionEngine(self.config.compiled_rules(), self.config.recheck_confidence)
        self.actuator = ActuatorController(self.config)
        self.edge_agent = EdgeAgent(self.decision_engine, self.actuator, self.config)
        self.integrated_agent = IntegratedAgentService(SQLITE_PATH, self.config, self.edge_agent)

        self.worker = DetectorWorker()
        self.worker.frame_signal.connect(self.show_frame)
        self.worker.result_signal.connect(self.show_result)
        self.worker.log_signal.connect(self.show_log)
        self.worker.stopped_signal.connect(self.on_stop)

        self.llm_worker: Optional[LLMAnalysisWorker] = None
        self.last_detection_payload: Optional[dict] = None
        self.last_frame = None
        self.last_saved_record: Optional[InspectionRecord] = None
        self.last_analysis = ""
        self.calibration_mode = False
        self.calibration_points: List[Tuple[int, int]] = []

        self.setWindowTitle(f"{APP_NAME} - {DEVICE_NAME}")
        self.setGeometry(60, 60, 1320, 860)

        self.apply_styles()
        self.init_ui()
        self.init_clock()
        self.apply_runtime_profile()
        self.update_stop_line_badge("normal", "正常监控")
        self.try_autoload_model()

    def rebuild_runtime_components(self) -> None:
        self.decision_engine = DecisionEngine(self.config.compiled_rules(), self.config.recheck_confidence)
        self.actuator = ActuatorController(self.config)
        self.edge_agent = EdgeAgent(self.decision_engine, self.actuator, self.config)
        self.integrated_agent = IntegratedAgentService(SQLITE_PATH, self.config, self.edge_agent)

    def save_station_config(self) -> None:
        self.config.station_name = self.station_name
        self.config.operator = self.station_operator
        self.config.default_model = self.default_model_path
        self.config.camera_index = self.camera_spin.value()
        self.config.confidence = round(self.conf_spin.value() / 100, 2)
        self.config.recheck_confidence = round(self.recheck_spin.value() / 100, 2)
        if hasattr(self, "confirm_spin"):
            self.config.confirm_frames = int(self.confirm_spin.value())
        if hasattr(self, "calibration_px_spin"):
            self.config.calibration_reference_px = int(self.calibration_px_spin.value())
        if hasattr(self, "calibration_mm_spin"):
            self.config.calibration_reference_mm = round(float(self.calibration_mm_spin.value()), 4)
            if self.config.calibration_reference_px > 0:
                self.config.mm_per_pixel = round(
                    self.config.calibration_reference_mm / self.config.calibration_reference_px,
                    6,
                )
        if hasattr(self, "ai_enable_toggle"):
            self.config.enable_ai_analysis = self.ai_enable_toggle.isChecked()
        if hasattr(self, "auto_ai_toggle"):
            self.config.auto_analyze_on_alert = self.auto_ai_toggle.isChecked()
        if hasattr(self, "llm_combo"):
            self.config.active_llm_model = self.llm_combo.currentData()
        self.config.save(CONFIG_PATH)
        self.rebuild_runtime_components()

    def refresh_quality_rules_editor(self) -> None:
        if not hasattr(self, "rules_table"):
            return
        ordered_names = sorted(self.config.quality_rules)
        self.rules_table.setRowCount(len(ordered_names))
        for row, defect_type in enumerate(ordered_names):
            rule = self.config.quality_rules[defect_type]

            name_item = QTableWidgetItem(defect_type)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.rules_table.setItem(row, 0, name_item)

            max_item = QTableWidgetItem(f"{float(rule.get('max_allowed_mm', DEFAULT_RULE_MAX_ALLOWED_MM)):.3f}")
            pass_item = QTableWidgetItem(f"{float(rule.get('pass_mm', DEFAULT_RULE_PASS_MM)):.3f}")
            zero_item = QTableWidgetItem()
            zero_item.setFlags(zero_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            zero_item.setCheckState(Qt.Checked if bool(rule.get("zero_tolerance", False)) else Qt.Unchecked)

            self.rules_table.setItem(row, 1, max_item)
            self.rules_table.setItem(row, 2, pass_item)
            self.rules_table.setItem(row, 3, zero_item)

        self.rules_table.resizeColumnsToContents()

    def add_quality_rule_row(self) -> None:
        name, accepted = QInputDialog.getText(self, "新增类别", "输入新的缺陷类别名:")
        if not accepted:
            return
        normalized = str(name).strip()
        if not normalized:
            QMessageBox.warning(self, "类别名无效", "类别名不能为空。")
            return
        if normalized in self.config.quality_rules:
            QMessageBox.warning(self, "类别名重复", f"类别 {normalized} 已存在。")
            return
        self.config.quality_rules[normalized] = default_quality_rule()
        self.save_station_config()
        self.refresh_quality_rules_editor()
        self.show_log(f"[规则] 已新增类别规则: {normalized}")

    def remove_selected_quality_rule_rows(self) -> None:
        if not hasattr(self, "rules_table"):
            return
        selected_rows = sorted({index.row() for index in self.rules_table.selectionModel().selectedRows()}, reverse=True)
        if not selected_rows:
            QMessageBox.information(self, "删除类别", "请先选中要删除的类别行。")
            return
        if len(self.config.quality_rules) - len(selected_rows) < 1:
            QMessageBox.warning(self, "删除类别", "至少保留一个缺陷类别规则。")
            return
        removed_names: List[str] = []
        for row in selected_rows:
            item = self.rules_table.item(row, 0)
            if not item:
                continue
            name = item.text().strip()
            if name and name in self.config.quality_rules:
                removed_names.append(name)
                self.config.quality_rules.pop(name, None)
        if not removed_names:
            return
        self.save_station_config()
        self.refresh_quality_rules_editor()
        self.show_log(f"[规则] 已删除类别规则: {', '.join(removed_names)}")

    def apply_quality_rules_editor(self) -> None:
        if not hasattr(self, "rules_table"):
            return

        normalized_rules: Dict[str, Dict[str, float]] = {}
        for row in range(self.rules_table.rowCount()):
            name_item = self.rules_table.item(row, 0)
            max_item = self.rules_table.item(row, 1)
            pass_item = self.rules_table.item(row, 2)
            zero_item = self.rules_table.item(row, 3)

            name = name_item.text().strip() if name_item else ""
            if not name:
                continue
            try:
                max_allowed_mm = float(max_item.text()) if max_item and max_item.text().strip() else DEFAULT_RULE_MAX_ALLOWED_MM
                pass_mm = float(pass_item.text()) if pass_item and pass_item.text().strip() else DEFAULT_RULE_PASS_MM
            except ValueError:
                QMessageBox.warning(self, "规则格式错误", f"类别 {name} 的数值字段无法转换。")
                return

            normalized_rules[name] = {
                "max_allowed_mm": max_allowed_mm,
                "pass_mm": pass_mm,
                "zero_tolerance": bool(zero_item and zero_item.checkState() == Qt.Checked),
            }

        if not normalized_rules:
            QMessageBox.warning(self, "规则格式错误", "至少保留一个缺陷类别规则。")
            return

        self.config.quality_rules = normalized_rules
        self.save_station_config()
        self.refresh_quality_rules_editor()
        self.show_log(f"[规则] 已应用 {len(normalized_rules)} 个类别的质检规则")

    def apply_runtime_profile(self) -> None:
        mm_per_pixel = self.config.resolved_mm_per_pixel()
        if self.config.runtime_profile == EDGE_RUNTIME_PROFILE:
            self.touch_hint.setText(f"边缘设备模式: 默认轻量输入、低频推理、AI 分析可选 | 标定 {mm_per_pixel:.4f} mm/px")
        else:
            self.touch_hint.setText(f"桌面模式: 可使用更高刷新率和更强本地分析能力 | 标定 {mm_per_pixel:.4f} mm/px")
        self.ai_model_badge.setText(f"AI模型 / {self.edge_agent.active_llm_label()}")
        if hasattr(self, "llm_combo"):
            current_index = self.llm_combo.findData(self.config.active_llm_model)
            if current_index >= 0:
                self.llm_combo.setCurrentIndex(current_index)
        if hasattr(self, "calibration_px_spin"):
            self.calibration_px_spin.setValue(int(self.config.calibration_reference_px))
        if hasattr(self, "calibration_mm_spin"):
            self.calibration_mm_spin.setValue(float(self.config.calibration_reference_mm))
        if hasattr(self, "ai_enable_toggle"):
            self.ai_enable_toggle.setChecked(self.config.enable_ai_analysis)
        if hasattr(self, "auto_ai_toggle"):
            self.auto_ai_toggle.setChecked(self.config.auto_analyze_on_alert)
        self.refresh_quality_rules_editor()

    def update_stop_line_badge(self, level: str, label: str) -> None:
        palette = {
            "normal": ("#0f2940", "#1d5075", "#dff2ff"),
            "warning": ("#5c4210", "#dba63d", "#fff6df"),
            "stop": ("#5f1922", "#db586b", "#fff0f2"),
        }
        background, border, color = palette.get(level, palette["normal"])
        self.stop_line_badge.setText(f"停线风险 / {label}")
        self.stop_line_badge.setStyleSheet(
            f"background:{background}; border:1px solid {border}; border-radius:12px; padding:9px 12px; color:{color}; font-weight:700;"
        )

    def available_llm_items(self) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for key, meta in self.config.llm_models.items():
            label = str(meta.get("label", key))
            enabled = bool(meta.get("enabled", False))
            items.append((key, f"{label}{'' if enabled else ' (disabled)'}"))
        return items

    def apply_llm_selection(self) -> None:
        selected_key = self.llm_combo.currentData()
        if not selected_key:
            return
        self.config.active_llm_model = selected_key
        self.save_station_config()
        self.apply_runtime_profile()
        self.show_log(f"[AI] 已切换辅助模型: {self.edge_agent.active_llm_label()}")

    def apply_ai_toggle(self) -> None:
        self.config.enable_ai_analysis = self.ai_enable_toggle.isChecked()
        self.save_station_config()
        self.apply_runtime_profile()
        self.show_log(f"[AI] 已{'启用' if self.config.enable_ai_analysis else '关闭'}本地辅助分析")

    def apply_auto_ai_toggle(self) -> None:
        self.config.auto_analyze_on_alert = self.auto_ai_toggle.isChecked()
        self.save_station_config()
        self.apply_runtime_profile()
        self.show_log(f"[AI] 已{'启用' if self.config.auto_analyze_on_alert else '关闭'}异常自动分析")

    def try_autoload_model(self) -> None:
        model_path = Path(self.default_model_path)
        if model_path.exists() and self.worker.load_model(str(model_path)):
            self.sync_quality_rules_with_model()
            self.model_label.setText(f"检测模型 / {model_path.name}")
            self.show_log(f"[系统] 已自动加载默认模型: {model_path.name}")

    def sync_quality_rules_with_model(self) -> None:
        if not self.worker.model:
            return
        added = self.config.ensure_quality_rules(extract_model_class_names(self.worker.model))
        if not added:
            return
        self.config.save(CONFIG_PATH)
        self.rebuild_runtime_components()
        self.refresh_quality_rules_editor()
        self.show_log(f"[规则] 已为新类别自动创建质检规则: {', '.join(added)}")

    def apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #06111b; color: #dfedf7; font-family: "Microsoft YaHei UI"; font-size: 14px; }
            QMainWindow { background: #06111b; }
            QFrame#topBanner, QFrame#panel { background: #0c1826; border: 1px solid #17344f; border-radius: 22px; }
            QLabel#pageTitle { font-size: 24px; font-weight: 700; color: #f3fbff; }
            QLabel#pageSubtitle { font-size: 13px; color: #86a9c2; }
            QLabel#clockLabel { font-size: 24px; font-weight: 700; color: #7ae7ff; }
            QLabel#dateLabel, QLabel#subtleLabel { color: #84a6c0; font-weight: 600; }
            QLabel#badge, QLabel#chip {
                background: #0f2940; border: 1px solid #1d5075; border-radius: 12px;
                padding: 9px 12px; color: #dff2ff; font-weight: 700;
            }
            QLabel#touchHint { color: #80a4bc; font-size: 12px; }
            QPushButton {
                min-height: 42px; background: #133553; color: #e6f6ff; border: 1px solid #22567a;
                border-radius: 12px; padding: 8px 14px; font-size: 14px; font-weight: 700;
            }
            QPushButton:hover { background: #194468; }
            QPushButton#accentButton { background: #0d7d71; border: 1px solid #18b4a4; }
            QPushButton#dangerButton { background: #6a1f2b; border: 1px solid #9b3347; }
            QPushButton#snapshotButton { background: #7a4e12; border: 1px solid #c58a28; }
            QRadioButton { font-size: 14px; font-weight: 700; color: #d1e5f5; spacing: 6px; }
            QRadioButton::indicator { width: 18px; height: 18px; }
            QSpinBox, QComboBox, QDoubleSpinBox {
                min-height: 38px; border-radius: 10px; border: 1px solid #1d5075;
                padding: 4px 8px; background: #091827; color: #e9f7ff; font-size: 14px; font-weight: 700;
            }
            QLabel#previewFrame {
                background: #04101a; border: 1px solid #21425f; border-radius: 24px;
                color: #86a9c2; font-size: 18px; font-weight: 700;
            }
            QLabel#heroStatus {
                background: #10273f; border: 1px solid #1d5075; border-radius: 18px;
                padding: 15px 18px; font-size: 15px; font-weight: 700;
            }
            QFrame#dataCard { background: #0f2032; border: 1px solid #1a4462; border-radius: 18px; }
            QLabel#cardKicker { color: #7fa5c1; font-size: 12px; font-weight: 700; }
            QLabel#cardValue { color: #f3fbff; font-size: 24px; font-weight: 700; }
            QLabel#cardDetail { color: #7fa5c1; font-size: 12px; }
            QTextEdit {
                background: #081421; border: 1px solid #17344f; border-radius: 16px;
                color: #deedf7; padding: 10px; selection-background-color: #0d7d71; font-size: 14px;
            }
            QTableWidget {
                background: #081421; border: 1px solid #17344f; border-radius: 16px;
                color: #deedf7; gridline-color: #17344f; font-size: 14px;
            }
            QHeaderView::section {
                background: #0f2940; color: #dff2ff; border: none; border-right: 1px solid #17344f;
                padding: 8px; font-weight: 700;
            }
            """
        )

    def init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        top_banner = QFrame()
        top_banner.setObjectName("topBanner")
        top_layout = QHBoxLayout(top_banner)
        top_layout.setContentsMargins(16, 14, 16, 14)
        top_layout.setSpacing(14)

        title_col = QVBoxLayout()
        title = QLabel("边缘AI工业质检工位")
        title.setObjectName("pageTitle")
        subtitle = QLabel("EdgeAgent 主控检测、规则判定、执行联动；本地模型仅用于辅助诊断")
        subtitle.setObjectName("pageSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        info_row = QHBoxLayout()
        self.station_badge = QLabel(f"工位 / {self.station_name}")
        self.station_badge.setObjectName("badge")
        self.operator_badge = QLabel(f"操作员 / {self.station_operator}")
        self.operator_badge.setObjectName("badge")
        self.ai_model_badge = QLabel(f"AI模型 / {self.edge_agent.active_llm_label()}")
        self.ai_model_badge.setObjectName("badge")
        self.stop_line_badge = QLabel("停线风险 / 正常监控")
        self.stop_line_badge.setObjectName("badge")
        info_row.addWidget(self.station_badge)
        info_row.addWidget(self.operator_badge)
        info_row.addWidget(self.ai_model_badge)
        info_row.addWidget(self.stop_line_badge)
        info_row.addStretch(1)
        title_col.addLayout(info_row)

        right_col = QVBoxLayout()
        right_col.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.clock_label = QLabel("--:--:--")
        self.clock_label.setObjectName("clockLabel")
        self.date_label = QLabel("--")
        self.date_label.setObjectName("dateLabel")
        self.touch_hint = QLabel("部署前请确认模型格式、规则库与执行器配置")
        self.touch_hint.setObjectName("touchHint")
        right_col.addWidget(self.clock_label, 0, Qt.AlignRight)
        right_col.addWidget(self.date_label, 0, Qt.AlignRight)
        right_col.addWidget(self.touch_hint, 0, Qt.AlignRight)

        top_layout.addLayout(title_col, 1)
        top_layout.addLayout(right_col)
        root.addWidget(top_banner)

        control_panel = QFrame()
        control_panel.setObjectName("panel")
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(14, 14, 14, 14)
        control_layout.setSpacing(10)

        top_info_row = QHBoxLayout()
        self.model_label = QLabel("检测模型 / 未加载")
        self.model_label.setObjectName("chip")
        self.source_label = QLabel("输入源 / 摄像头实时流")
        self.source_label.setObjectName("chip")
        self.storage_label = QLabel(f"留档目录 / {CAPTURE_DIR.name} | 运行档位 / {self.config.runtime_profile}")
        self.storage_label.setObjectName("chip")
        top_info_row.addWidget(self.model_label, 1)
        top_info_row.addWidget(self.source_label, 1)
        top_info_row.addWidget(self.storage_label, 1)
        control_layout.addLayout(top_info_row)

        source_row = QVBoxLayout()
        source_row.setSpacing(10)
        source_inputs_row = QHBoxLayout()
        source_inputs_row.setSpacing(8)
        source_actions_row = QHBoxLayout()
        source_actions_row.setSpacing(8)
        self.source_group = QButtonGroup()
        self.rb_cam = QRadioButton("摄像头")
        self.rb_video = QRadioButton("本地视频")
        self.rb_img = QRadioButton("本地图像")
        self.rb_cam.setChecked(True)
        self.source_group.addButton(self.rb_cam)
        self.source_group.addButton(self.rb_video)
        self.source_group.addButton(self.rb_img)

        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 5)
        self.camera_spin.setValue(int(self.config.camera_index))
        self.camera_spin.setPrefix("相机 ")

        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(5, 95)
        self.conf_spin.setValue(int(self.config.confidence * 100))
        self.conf_spin.setSuffix(" %")

        self.recheck_spin = QSpinBox()
        self.recheck_spin.setRange(10, 95)
        self.recheck_spin.setValue(int(self.config.recheck_confidence * 100))
        self.recheck_spin.setSuffix(" %复检")

        self.calibration_px_spin = QSpinBox()
        self.calibration_px_spin.setRange(1, 10000)
        self.calibration_px_spin.setValue(int(self.config.calibration_reference_px))
        self.calibration_px_spin.setPrefix("标定像素 ")

        self.calibration_mm_spin = QDoubleSpinBox()
        self.calibration_mm_spin.setRange(0.001, 1000.0)
        self.calibration_mm_spin.setDecimals(3)
        self.calibration_mm_spin.setSingleStep(0.01)
        self.calibration_mm_spin.setValue(float(self.config.calibration_reference_mm))
        self.calibration_mm_spin.setPrefix("标定毫米 ")

        self.ai_enable_toggle = QRadioButton("启用AI分析")
        self.ai_enable_toggle.setChecked(self.config.enable_ai_analysis)
        self.auto_ai_toggle = QRadioButton("异常自动分析")
        self.auto_ai_toggle.setChecked(self.config.auto_analyze_on_alert)

        self.confirm_spin = QSpinBox()
        self.confirm_spin.setRange(1, 10)
        self.confirm_spin.setValue(int(self.config.confirm_frames))
        self.confirm_spin.setPrefix("连续确认 ")
        self.confirm_spin.setSuffix(" 帧")

        self.llm_combo = QComboBox()
        self.llm_combo.setMinimumHeight(42)
        for key, label in self.available_llm_items():
            self.llm_combo.addItem(label, key)
        current_index = self.llm_combo.findData(self.config.active_llm_model)
        if current_index >= 0:
            self.llm_combo.setCurrentIndex(current_index)

        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_export_onnx = QPushButton("导出 ONNX")
        self.btn_export_onnx.clicked.connect(self.export_onnx_model)
        self.btn_export_rknn = QPushButton("导出 RKNN")
        self.btn_export_rknn.clicked.connect(self.export_rknn_model)
        self.btn_select_file = QPushButton("选择文件")
        self.btn_select_file.clicked.connect(self.select_file)
        self.btn_apply_ai = QPushButton("应用AI开关")
        self.btn_apply_ai.clicked.connect(self.apply_ai_toggle)
        self.btn_apply_auto_ai = QPushButton("应用自动分析")
        self.btn_apply_auto_ai.clicked.connect(self.apply_auto_ai_toggle)
        self.btn_apply_llm = QPushButton("切换AI模型")
        self.btn_apply_llm.clicked.connect(self.apply_llm_selection)
        self.btn_start = QPushButton("启动检测")
        self.btn_start.setObjectName("accentButton")
        self.btn_start.clicked.connect(self.start_detect)
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.setObjectName("dangerButton")
        self.btn_stop.clicked.connect(self.stop_detect)
        self.btn_snapshot = QPushButton("保存快照")
        self.btn_snapshot.setObjectName("snapshotButton")
        self.btn_snapshot.clicked.connect(self.save_manual_snapshot)
        self.btn_ai = QPushButton("AI 辅助分析")
        self.btn_ai.clicked.connect(self.start_ai_analysis)
        self.btn_review = QPushButton("人工复判")
        self.btn_review.clicked.connect(self.open_human_review_dialog)
        self.btn_shift_summary = QPushButton("班次总结")
        self.btn_shift_summary.clicked.connect(self.show_shift_summary)
        self.btn_rule_suggestions = QPushButton("规则建议")
        self.btn_rule_suggestions.clicked.connect(self.show_rule_suggestions)
        self.btn_calibrate = QPushButton("开始标定")
        self.btn_calibrate.clicked.connect(self.toggle_calibration_mode)
        self.btn_clear_calibration = QPushButton("清空标定点")
        self.btn_clear_calibration.clicked.connect(self.clear_calibration_points)

        for widget in (
            self.rb_cam,
            self.rb_video,
            self.rb_img,
            self.camera_spin,
            self.conf_spin,
            self.recheck_spin,
            self.calibration_px_spin,
            self.calibration_mm_spin,
            self.confirm_spin,
            self.ai_enable_toggle,
            self.auto_ai_toggle,
            self.llm_combo,
        ):
            source_inputs_row.addWidget(widget)
        source_inputs_row.addStretch(1)
        for widget in (
            self.btn_load_model,
            self.btn_export_onnx,
            self.btn_export_rknn,
            self.btn_select_file,
            self.btn_apply_ai,
            self.btn_apply_auto_ai,
            self.btn_apply_llm,
            self.btn_start,
            self.btn_stop,
            self.btn_snapshot,
            self.btn_ai,
            self.btn_review,
            self.btn_shift_summary,
            self.btn_rule_suggestions,
            self.btn_calibrate,
            self.btn_clear_calibration,
        ):
            source_actions_row.addWidget(widget)
        source_actions_row.addStretch(1)
        source_row.addLayout(source_inputs_row)
        source_row.addLayout(source_actions_row)
        control_layout.addLayout(source_row)
        root.addWidget(control_panel)

        content_row = QHBoxLayout()

        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)

        preview_header = QHBoxLayout()
        preview_title = QLabel("检测画面")
        preview_title.setObjectName("pageSubtitle")
        preview_title.setStyleSheet("font-size:16px; color:#e6f6ff; font-weight:700;")
        self.preview_meta = QLabel("设备状态 / 待机")
        self.preview_meta.setObjectName("subtleLabel")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.preview_meta)
        left_layout.addLayout(preview_header)

        self.img_label = CalibrationPreviewLabel("等待接入检测画面")
        self.img_label.setObjectName("previewFrame")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(680, 420)
        self.img_label.point_selected.connect(self.handle_calibration_point)
        left_layout.addWidget(self.img_label, 1)

        self.status_label = QLabel("状态 / 待机 | 缺陷 / 0 | 判定 / 未执行 | 动作 / 未执行")
        self.status_label.setObjectName("heroStatus")
        left_layout.addWidget(self.status_label)
        content_row.addWidget(left_panel, 2)

        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)

        metrics_title = QLabel("工位指标")
        metrics_title.setObjectName("pageSubtitle")
        metrics_title.setStyleSheet("font-size:16px; color:#e6f6ff; font-weight:700;")
        right_layout.addWidget(metrics_title)

        metric_grid = QGridLayout()
        self.status_card = DataCard("当前判定", "待机", "#38bdf8")
        self.defect_card = DataCard("当前缺陷数", "0", "#ef4444")
        self.conf_card = DataCard("最高置信度", "0.00", "#f59e0b")
        self.class_card = DataCard("缺陷类别", "暂无", "#22c55e")
        self.total_card = DataCard("累计检测数", "0", "#60a5fa")
        self.yield_card = DataCard("放行率", "0.0%", "#34d399")
        metric_grid.addWidget(self.status_card, 0, 0)
        metric_grid.addWidget(self.defect_card, 0, 1)
        metric_grid.addWidget(self.conf_card, 1, 0)
        metric_grid.addWidget(self.class_card, 1, 1)
        metric_grid.addWidget(self.total_card, 2, 0)
        metric_grid.addWidget(self.yield_card, 2, 1)
        right_layout.addLayout(metric_grid)

        rules_title = QLabel("质检规则")
        rules_title.setObjectName("pageSubtitle")
        rules_title.setStyleSheet("font-size:16px; color:#e6f6ff; font-weight:700;")
        right_layout.addWidget(rules_title)

        rules_toolbar = QHBoxLayout()
        self.btn_add_rule = QPushButton("新增类别")
        self.btn_add_rule.clicked.connect(self.add_quality_rule_row)
        self.btn_remove_rule = QPushButton("删除选中")
        self.btn_remove_rule.clicked.connect(self.remove_selected_quality_rule_rows)
        self.btn_refresh_rules = QPushButton("刷新规则")
        self.btn_refresh_rules.clicked.connect(self.refresh_quality_rules_editor)
        self.btn_apply_rules = QPushButton("应用规则")
        self.btn_apply_rules.clicked.connect(self.apply_quality_rules_editor)
        rules_toolbar.addWidget(self.btn_add_rule)
        rules_toolbar.addWidget(self.btn_remove_rule)
        rules_toolbar.addWidget(self.btn_refresh_rules)
        rules_toolbar.addWidget(self.btn_apply_rules)
        rules_toolbar.addStretch(1)
        right_layout.addLayout(rules_toolbar)

        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(4)
        self.rules_table.setHorizontalHeaderLabels(["类别", "max_allowed_mm", "pass_mm", "zero_tolerance"])
        self.rules_table.verticalHeader().setVisible(False)
        self.rules_table.setAlternatingRowColors(True)
        self.rules_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rules_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.rules_table.setMinimumHeight(220)
        right_layout.addWidget(self.rules_table, 1)
        self.refresh_quality_rules_editor()

        log_title = QLabel("运行日志")
        log_title.setObjectName("pageSubtitle")
        log_title.setStyleSheet("font-size:16px; color:#e6f6ff; font-weight:700;")
        right_layout.addWidget(log_title)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        right_layout.addWidget(self.log_text, 1)

        ai_title = QLabel("AI 辅助建议")
        ai_title.setObjectName("pageSubtitle")
        ai_title.setStyleSheet("font-size:16px; color:#e6f6ff; font-weight:700;")
        right_layout.addWidget(ai_title)
        self.ai_text = QTextEdit()
        self.ai_text.setReadOnly(True)
        self.ai_text.setMinimumHeight(240)
        self.ai_text.setPlaceholderText("你可以在界面或 station_config.json 中切换本地模型。")
        right_layout.addWidget(self.ai_text, 1)
        content_row.addWidget(right_panel, 1)
        root.addLayout(content_row, 1)

    def init_clock(self) -> None:
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

    def update_clock(self) -> None:
        now = QDateTime.currentDateTime()
        self.clock_label.setText(now.toString("hh:mm:ss"))
        self.date_label.setText(now.toString("yyyy-MM-dd dddd"))

    def load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "Model (*.pt *.onnx *.rknn)")
        if path and self.worker.load_model(path):
            self.default_model_path = path
            self.sync_quality_rules_with_model()
            self.model_label.setText(f"检测模型 / {Path(path).name}")
            self.show_log(f"[系统] 已切换模型: {path}")
            self.save_station_config()

    def export_onnx_model(self) -> None:
        if not self.worker.model:
            QMessageBox.warning(self, "导出 ONNX", "请先加载 .pt 模型。")
            return

        model_name = Path(self.worker.current_model_path).stem if self.worker.current_model_path else "model"
        default_path = str((EXPORT_DIR / f"{model_name}.onnx").resolve())
        path, _ = QFileDialog.getSaveFileName(self, "导出 ONNX", default_path, "ONNX (*.onnx)")
        if not path:
            return

        self.show_log("[导出] 开始导出 ONNX...")
        success, message = self.worker.export_model("onnx", path)
        if success:
            self.show_log(f"[导出] ONNX 导出完成: {message}")
            QMessageBox.information(self, "导出 ONNX", f"ONNX 已导出到:\n{message}")
            return

        self.show_log(f"[导出] ONNX 导出失败: {message}")
        QMessageBox.warning(self, "导出 ONNX", message)

    def export_rknn_model(self) -> None:
        if not self.worker.model:
            QMessageBox.warning(self, "导出 RKNN", "请先加载 .pt 模型。")
            return

        target, accepted = QInputDialog.getItem(
            self,
            "RKNN 平台",
            "选择目标平台",
            RKNN_TARGET_PLATFORMS,
            0,
            False,
        )
        if not accepted or not target:
            return

        model_name = Path(self.worker.current_model_path).stem if self.worker.current_model_path else "model"
        path = QFileDialog.getExistingDirectory(self, "选择 RKNN 输出目录", str(EXPORT_DIR.resolve()))
        if not path:
            return

        target_dir = str((Path(path) / f"{model_name}_{target}_rknn_model").resolve())
        self.show_log(f"[导出] 开始导出 RKNN，target={target} ...")
        success, message = self.worker.export_model("rknn", target_dir, target)
        if success:
            self.show_log(f"[导出] RKNN 导出完成: {message}")
            QMessageBox.information(self, "导出 RKNN", f"RKNN 已导出到:\n{message}")
            return

        self.show_log(f"[导出] RKNN 导出失败: {message}")
        QMessageBox.warning(self, "导出 RKNN", message)

    def select_file(self) -> None:
        if self.rb_video.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video (*.mp4 *.avi *.mov)")
        elif self.rb_img.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image (*.jpg *.jpeg *.png *.bmp)")
        else:
            self.source_label.setText("输入源 / 摄像头实时流")
            return

        if path:
            self.worker.source_path = path
            self.source_label.setText(f"输入源 / {Path(path).name}")
            self.show_log(f"[选择] {path}")

    def start_detect(self) -> None:
        if self.worker.isRunning():
            self.show_log("[提示] 检测线程正在运行。")
            return
        if not self.worker.model:
            QMessageBox.warning(self, "提示", "请先加载检测模型。")
            return

        self.worker.set_camera_index(self.camera_spin.value())
        self.worker.set_confidence(self.conf_spin.value() / 100)
        self.worker.set_recheck_confidence(self.recheck_spin.value() / 100)
        self.worker.set_runtime(
            self.config.runtime_profile,
            self.config.input_width,
            self.config.input_height,
            self.config.frame_interval_ms,
            self.config.confirm_frames,
            self.config.resolved_mm_per_pixel(),
        )

        if self.rb_cam.isChecked():
            self.worker.set_source("camera")
            self.source_label.setText(f"输入源 / 摄像头 {self.camera_spin.value()}")
        elif self.rb_video.isChecked():
            if not self.worker.source_path:
                QMessageBox.warning(self, "提示", "请先选择视频文件。")
                return
            self.worker.set_source("video", self.worker.source_path)
        else:
            if not self.worker.source_path:
                QMessageBox.warning(self, "提示", "请先选择图像文件。")
                return
            self.worker.set_source("image", self.worker.source_path)

        self.preview_meta.setText("设备状态 / 检测中")
        self.status_card.set_value("运行中")
        self.status_card.set_detail("检测 -> EdgeAgent -> 留档")
        self.status_label.setStyleSheet(
            "background:#0d4168; border:1px solid #2b8fca; color:#effbff; border-radius:18px; padding:15px 18px; font-size:15px; font-weight:700;"
        )
        self.status_label.setText("状态 / 检测中 | 缺陷 / 0 | 判定 / 未执行 | 动作 / 未执行")
        self.ai_text.clear()
        self.last_analysis = ""
        self.save_station_config()
        self.apply_runtime_profile()
        self.show_log(
            f"[系统] 检测任务已启动。profile={self.config.runtime_profile} input={self.config.input_width}x{self.config.input_height} interval={self.config.frame_interval_ms}ms"
        )
        self.worker.start()

    def stop_detect(self) -> None:
        if self.worker.isRunning():
            self.worker.stop()
            self.show_log("[系统] 检测任务已停止。")

    def on_stop(self) -> None:
        self.preview_meta.setText("设备状态 / 已停止")
        self.status_card.set_value("已停止")
        self.status_card.set_detail("等待下一次启动")
        self.status_label.setStyleSheet(
            "background:#1c2e42; border:1px solid #37597a; color:#e5f3ff; border-radius:18px; padding:15px 18px; font-size:15px; font-weight:700;"
        )
        self.status_label.setText("状态 / 已停止")

    def show_frame(self, image: QImage) -> None:
        self.img_label.set_frame_image(image)

    def toggle_calibration_mode(self) -> None:
        self.calibration_mode = not self.calibration_mode
        self.img_label.set_calibration_enabled(self.calibration_mode)
        if self.calibration_mode:
            self.calibration_points = []
            self.img_label.set_calibration_points(self.calibration_points)
            self.preview_meta.setText("设备状态 / 标定中，请在画面上点击两点")
            self.btn_calibrate.setText("取消标定")
            self.show_log("[标定] 已进入标定模式，请在预览图上选择两点。")
        else:
            self.calibration_points = []
            self.img_label.set_calibration_points(self.calibration_points)
            self.preview_meta.setText("设备状态 / 待机")
            self.btn_calibrate.setText("开始标定")
            self.show_log("[标定] 已退出标定模式。")

    def clear_calibration_points(self) -> None:
        self.calibration_points = []
        self.img_label.set_calibration_points(self.calibration_points)
        self.show_log("[标定] 已清空当前标定点。")
        if self.calibration_mode:
            self.preview_meta.setText("设备状态 / 标定中，请重新点击两点")

    def handle_calibration_point(self, x: int, y: int) -> None:
        if not self.calibration_mode:
            return
        self.calibration_points.append((x, y))
        self.img_label.set_calibration_points(self.calibration_points)
        self.show_log(f"[标定] 记录点 {len(self.calibration_points)}: ({x}, {y})")
        if len(self.calibration_points) < 2:
            self.preview_meta.setText("设备状态 / 标定中，请选择第 2 个点")
            return

        point_a = self.calibration_points[0]
        point_b = self.calibration_points[1]
        distance_px = ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5
        self.calibration_px_spin.setValue(max(1, int(round(distance_px))))
        self.save_station_config()
        self.apply_runtime_profile()
        self.show_log(f"[标定] 两点像素距离={distance_px:.2f}px，已更新 Calibration px。")
        self.calibration_mode = False
        self.img_label.set_calibration_enabled(False)
        self.preview_meta.setText("设备状态 / 标定完成")
        self.btn_calibrate.setText("开始标定")

    def save_inspection_record(self, decision: InspectionDecision, payload: dict) -> InspectionRecord:
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
        source_name = "camera" if self.rb_cam.isChecked() else Path(self.worker.source_path).name
        image_name = f"{timestamp}_{decision.decision}.jpg"
        image_path = CAPTURE_DIR / image_name
        cv2.imwrite(str(image_path), payload["annotated_frame"])

        record = InspectionRecord(
            record_id=uuid.uuid4().hex,
            timestamp=QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
            station=self.station_name,
            operator=self.station_operator,
            source_type=self.worker.source_type,
            source_name=source_name,
            detector_status=decision.detector_status,
            decision=decision.decision,
            action=decision.action,
            defect_count=payload["defect_count"],
            top_confidence=payload["top_confidence"],
            avg_confidence=payload["avg_confidence"],
            classes=payload["classes"],
            inference_ms=payload["inference_ms"],
            latency_ms=payload["latency_ms"],
            image_path=str(image_path),
            notes=decision.notes,
            analysis=self.last_analysis,
        )

        with CSV_PATH.open("a", newline="", encoding="utf-8-sig") as file:
            csv.writer(file).writerow(record.to_row())
        with JSONL_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        self.insert_inspection_record(record)

        self.show_log(f"[留档] 已保存检测记录: {image_name}")
        return record

    def insert_inspection_record(self, record: InspectionRecord) -> None:
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO inspection_records (
                    record_id, timestamp, station, operator, source_type, source_name,
                    detector_status, decision, action, defect_count, top_confidence,
                    avg_confidence, classes_json, inference_ms, latency_ms,
                    image_path, notes, analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record.to_db_row(),
            )
            conn.commit()

    def show_result(self, payload: dict) -> None:
        if not payload.get("confirmed", True):
            class_text = "、".join(payload["classes"]) if payload["classes"] else "暂无"
            confirmed_frames = self.config.confirm_frames - int(payload.get("pending_frames", 0))
            self.last_frame = payload["annotated_frame"]
            self.last_detection_payload = payload
            self.status_card.set_value("PENDING")
            self.status_card.set_detail(f"等待连续确认 {confirmed_frames}/{self.config.confirm_frames}")
            self.defect_card.set_value(str(payload.get("raw_defect_count", 0)))
            self.defect_card.set_detail("当前结果尚未达到连续帧阈值")
            self.conf_card.set_value(f"{payload['top_confidence']:.2f}")
            self.conf_card.set_detail(f"均值 {payload['avg_confidence']:.2f}")
            self.class_card.set_value(class_text)
            self.class_card.set_detail(
                f"推理 {payload['inference_ms']:.1f} ms | 估算 {payload.get('max_estimated_size_mm', 0.0):.3f} mm"
            )
            self.preview_meta.setText("设备状态 / 待确认")
            self.status_label.setStyleSheet(
                "background:#364152; border:1px solid #6b7280; color:#f7fbff; border-radius:18px; padding:15px 18px; font-size:15px; font-weight:700;"
            )
            self.status_label.setText(
                f"状态 / PENDING | 连续确认 / {confirmed_frames}/{self.config.confirm_frames} | 候选缺陷 / {payload.get('raw_defect_count', 0)} | 类别 / {class_text}"
            )
            self.ai_text.setPlainText("当前异常尚未通过连续帧确认，暂不触发 AI 分析。")
            return

        decision, action_detail = self.edge_agent.process_detection(payload)
        class_text = "、".join(payload["classes"]) if payload["classes"] else "暂无"

        self.last_frame = payload["annotated_frame"]
        self.last_detection_payload = payload

        self.status_card.set_value(decision.decision)
        self.status_card.set_detail(decision.notes)
        self.defect_card.set_value(str(payload["defect_count"]))
        self.defect_card.set_detail(f"检测状态 {payload['detector_status']}")
        self.conf_card.set_value(f"{payload['top_confidence']:.2f}")
        self.conf_card.set_detail(f"均值 {payload['avg_confidence']:.2f}")
        self.class_card.set_value(class_text)
        self.class_card.set_detail(
            f"推理 {payload['inference_ms']:.1f} ms | 估算 {payload.get('max_estimated_size_mm', 0.0):.3f} mm"
        )
        self.total_card.set_value(str(self.edge_agent.state.total_count))
        self.total_card.set_detail(
            f"OK {self.edge_agent.state.ok_count} / RECHECK {self.edge_agent.state.recheck_count} / NG {self.edge_agent.state.ng_count}"
        )
        self.yield_card.set_value(f"{self.edge_agent.pass_rate():.1f}%")
        self.yield_card.set_detail(f"端到端 {payload['latency_ms']:.1f} ms")

        if decision.decision == "OK":
            color, border, status_text = "#0e4d3f", "#2dbd8a", "设备状态 / 正常"
        elif decision.decision == "RECHECK":
            color, border, status_text = "#6a4d14", "#dba63d", "设备状态 / 待复检"
        else:
            color, border, status_text = "#5f1922", "#db586b", "设备状态 / 告警"
        self.preview_meta.setText(status_text)

        self.status_label.setStyleSheet(
            f"background:{color}; border:1px solid {border}; color:#f7fbff; border-radius:18px; padding:15px 18px; font-size:15px; font-weight:700;"
        )
        self.status_label.setText(
            f"状态 / {payload['detector_status']} | 判定 / {decision.decision} | 动作 / {decision.action} | 缺陷 / {payload['defect_count']} | 类别 / {class_text} | 尺寸 / {payload.get('max_estimated_size_mm', 0.0):.3f} mm"
        )

        if decision.decision in {"NG", "RECHECK"}:
            self.last_saved_record = self.save_inspection_record(decision, payload)
        else:
            self.last_saved_record = None
        self.show_log(f"[执行] {action_detail}")

        stop_line_advice_raw = self.integrated_agent.run(
            "stop_line_advice",
            decision=decision.decision,
            action=decision.action,
            consecutive_ng=self.actuator.consecutive_ng,
            stop_line_threshold=self.config.stop_line_ng_threshold,
            classes=payload.get("classes", []),
        )
        stop_line_advice = json.loads(stop_line_advice_raw)
        stop_line_message = str(stop_line_advice.get("message", ""))
        stop_line_level = str(stop_line_advice.get("level", "normal"))
        stop_line_label = str(stop_line_advice.get("label", "正常监控"))
        self.update_stop_line_badge(stop_line_level, stop_line_label)
        self.show_log(f"[Agent] {stop_line_message}")

        if decision.decision == "OK":
            self.ai_text.setPlainText(f"当前结果已放行，无需额外辅助分析。\n{stop_line_message}")
        elif self.edge_agent.should_trigger_ai(decision):
            self.ai_text.setPlainText(stop_line_message)
            self.start_ai_analysis(auto_trigger=True)
        else:
            self.ai_text.setPlainText(stop_line_message)

    def update_last_record_analysis(self, analysis_text: str) -> None:
        self.last_analysis = analysis_text
        if not self.last_saved_record:
            return
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.execute(
                    "UPDATE inspection_records SET analysis = ? WHERE record_id = ?",
                    (analysis_text, self.last_saved_record.record_id),
                )
                conn.commit()
        except Exception:
            pass

    def save_human_review(self, record_id: str, human_label: str, review_note: str) -> None:
        self.integrated_agent.run(
            "human_review",
            record_id=record_id,
            human_label=human_label,
            review_note=review_note,
        )

    def open_human_review_dialog(self) -> None:
        if not self.last_saved_record:
            QMessageBox.information(self, "人工复判", "当前没有可复判的异常记录。")
            return

        review_options = ["OK放行", "误报", "漏报", "NG确认", "人工复检"]
        selected_label, accepted = QInputDialog.getItem(
            self,
            "人工复判",
            "请选择人工复判结果",
            review_options,
            0,
            False,
        )
        if not accepted or not selected_label:
            return

        review_note, note_accepted = QInputDialog.getText(
            self,
            "人工复判备注",
            "可选填写备注",
        )
        if not note_accepted:
            return

        cleaned_note = review_note.strip()
        review_result = self.integrated_agent.run(
            "human_review",
            record_id=self.last_saved_record.record_id,
            human_label=selected_label,
            review_note=cleaned_note,
        )
        self.show_log(f"[复判] 已保存人工复判: {selected_label}")
        self.ai_text.setPlainText(review_result)

    def show_shift_summary(self) -> None:
        summary = self.integrated_agent.run("shift_summary")
        self.ai_text.setPlainText(summary)
        self.show_log("[Agent] 已生成班次总结。")

    def show_rule_suggestions(self) -> None:
        suggestions = self.integrated_agent.run("rule_suggestions")
        self.ai_text.setPlainText(suggestions)
        self.show_log("[Agent] 已生成规则建议。")

    def save_manual_snapshot(self) -> None:
        if self.last_frame is None:
            QMessageBox.information(self, "提示", "当前没有可保存的检测画面。")
            return
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss_zzz")
        path = CAPTURE_DIR / f"{timestamp}_manual.jpg"
        cv2.imwrite(str(path), self.last_frame)
        self.show_log(f"[快照] 已保存人工快照: {path.name}")
        QMessageBox.information(self, "快照完成", f"快照已保存到:\n{path}")

    def build_analysis_prompt(self) -> str:
        return self.integrated_agent.run(
            "analysis_prompt",
            payload=self.last_detection_payload,
            station_name=self.station_name,
            station_operator=self.station_operator,
        )

    def start_ai_analysis(self, auto_trigger: bool = False) -> None:
        reason = self.edge_agent.ai_unavailable_reason()
        if reason:
            self.ai_text.setPlainText(reason)
            return
        if self.last_detection_payload is None:
            self.ai_text.setPlainText("暂无检测结果，无法进行辅助分析。")
            return
        if self.llm_worker and self.llm_worker.isRunning():
            return

        prompt = self.build_analysis_prompt()
        self.ai_text.setPlainText(f"AI 辅助分析中，请稍候...\n当前模型: {self.edge_agent.active_llm_label()}")
        self.llm_worker = LLMAnalysisWorker(
            prompt=prompt,
            api_url=self.config.llm_api_url,
            model_name=self.edge_agent.active_llm_model_name(),
        )
        self.llm_worker.result_signal.connect(lambda text, auto=auto_trigger: self.show_ai_result(text, auto))
        self.llm_worker.start()
        self.show_log(f"[AI] 已提交辅助分析请求，模型={self.edge_agent.active_llm_model_name()}")

    def show_ai_result(self, result: str, auto_trigger: bool = False) -> None:
        current_text = self.ai_text.toPlainText().strip()
        if current_text and current_text != result:
            self.ai_text.setPlainText(f"{current_text}\n\n{result}")
        else:
            self.ai_text.setPlainText(result)
        self.update_last_record_analysis(result)
        self.show_log(f"[AI] {'自动触发分析' if auto_trigger else '人工触发分析'}完成。")

    def show_log(self, message: str) -> None:
        current = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_text.append(f"[{current}] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        if isinstance(scrollbar, QScrollBar):
            scrollbar.setValue(scrollbar.maximum())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Q:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.actuator.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    if window.config.auto_fullscreen:
        window.showFullScreen()
    else:
        window.show()
    sys.exit(app.exec_())

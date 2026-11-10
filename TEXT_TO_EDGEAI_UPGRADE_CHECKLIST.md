# `D:\text\agy-llm.py` 升级到 `D:\edgeai\agy-llm.py` 变更清单

## 目标

将 `D:\text\agy-llm.py` 的基础版能力，升级到 `D:\edgeai\agy-llm.py` 当前版本所具备的增强能力，包括：

- 结构化存储
- 连续帧确认
- 标定与像素转毫米
- 人工复判闭环
- 集成式 agent
- 停线建议与风险展示
- 更紧凑的界面布局

---

## 1. 依赖与导入

需要新增的 Python 模块：

- `sqlite3`
- `uuid`

需要新增的 Qt / GUI 相关导入：

- `QPoint`
- `QDoubleSpinBox`
- `QColor`
- `QPainter`
- `QPen`

目的：

- 支持 SQLite 持久化
- 支持记录唯一 ID
- 支持交互式标定
- 支持预览图绘制标定点和连线

---

## 2. 配置模型扩展

### `StationConfig` 新增字段

- `auto_fullscreen = False`
- `confirm_frames`
- `calibration_reference_px`
- `calibration_reference_mm`
- `mm_per_pixel`

### 新增方法

- `resolved_mm_per_pixel()`

作用：

- 统一管理连续帧确认参数
- 统一管理像素转毫米标定参数
- 启动时自动推导当前有效换算比

### 配置加载兼容逻辑

`load()` 中需要增加：

- 如果配置里存在 `calibration_reference_px`
- 且存在 `calibration_reference_mm`
- 自动推导 `mm_per_pixel`

---

## 3. 存储层升级

### 新增路径常量

- `SQLITE_PATH = REPORT_DIR / "inspection_records.db"`

### 新增方法

- `ensure_sqlite_db()`

### SQLite 表

#### `inspection_records`

字段建议：

- `record_id`
- `timestamp`
- `station`
- `operator`
- `source_type`
- `source_name`
- `detector_status`
- `decision`
- `action`
- `defect_count`
- `top_confidence`
- `avg_confidence`
- `classes_json`
- `inference_ms`
- `latency_ms`
- `image_path`
- `notes`
- `analysis`

#### `human_reviews`

字段建议：

- `review_id`
- `record_id`
- `human_label`
- `review_note`
- `reviewed_at`

### `InspectionRecord` 扩展

新增：

- `record_id`
- `to_db_row()`

### CSV 兼容迁移

需要把旧 CSV 表头：

- 从无 `record_id`

升级为：

- 第一列新增 `record_id`

并兼容旧数据自动补空列。

---

## 4. 检测链路升级

### `DetectorWorker` 新增字段

- `confirm_frames`
- `mm_per_pixel`
- `detection_streak`
- `last_detection_signature`

### 新增方法

- `estimate_size_mm()`

### 修改逻辑

#### 尺寸估算

旧版：

- 固定 `pixel / 100.0`

新版：

- `pixel_span * mm_per_pixel`

#### 连续帧确认

新增机制：

- 单帧命中先不直接进入有效异常
- 连续达到 `confirm_frames` 后才算确认异常

#### 新状态

- `PENDING`

#### 新 payload 字段

- `raw_boxes`
- `raw_defect_count`
- `confirmed`
- `pending_frames`
- `max_estimated_size_mm`
- `mm_per_pixel`

---

## 5. 标定功能

### 新增 UI 控件

- `calibration_px_spin`
- `calibration_mm_spin`

### 新增交互式预览控件

- `CalibrationPreviewLabel`

能力：

- 点击两点取像素距离
- 在预览图上显示标定点
- 显示两点连线
- 显示像素距离标签

### 新增方法

- `toggle_calibration_mode()`
- `clear_calibration_points()`
- `handle_calibration_point()`

### 标定流程

1. 开启标定模式
2. 在画面上点两点
3. 自动计算像素距离
4. 回填 `Calibration px`
5. 根据 `Calibration mm` 计算 `mm_per_pixel`

---

## 6. 记录与复判闭环

### 新增数据库写入

- `insert_inspection_record()`

### 分析结果回写升级

旧版：

- 通过“重写 CSV 最后一行”的方式更新分析结果

新版：

- 使用 `record_id`
- 按记录精确更新 SQLite 中对应行

### 新增人工复判

新增方法：

- `save_human_review()`
- `open_human_review_dialog()`

新增 UI 按钮：

- `人工复判`

作用：

- 为后续规则建议提供真实人工反馈

---

## 7. Agent 能力升级

### 新增服务类

- `IntegratedAgentService`

### 服务能力

- `build_analysis_prompt()`
- `build_shift_summary()`
- `build_rule_suggestions()`
- `record_human_review()`
- `should_escalate_stop_line()`
- `run(action, **kwargs)`

### 集成入口

UI 和主流程不直接散调各方法，而是统一走：

- `integrated_agent.run(...)`

### 已接入的动作

- `analysis_prompt`
- `shift_summary`
- `rule_suggestions`
- `human_review`
- `stop_line_advice`

---

## 8. 停线建议与风险展示

### 新增能力

- `should_escalate_stop_line()`

### 判断输入

- 当前 `decision`
- 当前 `action`
- `consecutive_ng`
- `stop_line_ng_threshold`
- 最近同类异常比例

### 风险级别

- `normal`
- `warning`
- `stop`

### UI 新增

- `stop_line_badge`

显示文案：

- `停线风险 / 正常监控`
- `停线风险 / 接近停线`
- `停线风险 / 建议停线`

### 行为说明

当前版本：

- 只做建议
- 不直接替代执行器动作

---

## 9. 班次总结与规则建议

### 新增按钮

- `班次总结`
- `规则建议`

### 班次总结内容

- 最近检测记录数量
- 判定分布
- 缺陷总数
- 平均最高置信度
- 缺陷 Top5
- 人工复判分布

### 规则建议内容

结合 `human_reviews` 判断：

- 哪类缺陷误报偏多
- 哪类缺陷异常稳定
- 哪类缺陷人工复检比例偏高

输出建议如：

- 提高置信度阈值
- 收紧 `pass_mm`
- 收紧 `max_allowed_mm`
- 增加连续帧确认

---

## 10. 界面与布局优化

### 默认行为调整

- 默认不全屏
- 默认窗口尺寸缩小

### 顶部控制区优化

- 从一整行改为两行
- 输入控件一行
- 操作按钮一行

### 尺寸调整

- 按钮高度减小
- 下拉框高度减小
- 字号和边距收紧
- 预览区最小尺寸减小

### 中文化

将英文控件文本替换为中文，例如：

- `Calibration px` -> `标定像素`
- `Calibration mm` -> `标定毫米`
- `Confirm frames` -> `连续确认`

---

## 推荐升级顺序

建议按以下顺序迁移：

1. 先加 `sqlite + record_id`
2. 再加 `confirm_frames`
3. 再加标定与 `mm_per_pixel`
4. 再加人工复判
5. 再加 `IntegratedAgentService`
6. 再加班次总结、规则建议
7. 最后加停线建议与停线风险展示

---

## 风险提示

不建议一步到位直接把所有功能一次性搬过去，原因：

- 基础版没有结构化数据底座
- 一次性改动过大，不利于定位问题
- UI、检测链、存储链、agent 链混改风险高

推荐做法：

- 每完成一组功能，就做一次运行验证
- 每完成一组功能，就做一次最小可用测试

---

## 迁移完成后的目标状态

迁移完成后，`D:\text\agy-llm.py` 应具备和 `D:\edgeai\agy-llm.py` 同等级能力：

- 结构化检测记录
- 交互式标定
- 连续帧确认
- 人工复判闭环
- 集成式 agent
- 班次总结
- 规则建议
- 停线建议
- 风险级别可视化

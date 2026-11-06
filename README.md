# Blender Cloud Render Farm UI on Google Colab

เรนเดอร์ไฟล์ `.blend` ด้วย GPU บน Google Colab ผ่านหน้าเว็บ Gradio พร้อมบันทึกผลลัพธ์ลง Google Drive
A web UI for rendering `.blend` files on a Colab GPU with Gradio, saving frames to Google Drive as they finish.

---

## ภาษาไทย

### คุณสมบัติ
- อัปโหลดไฟล์ `.blend` หรือเลือกจาก `Google Drive/MyDrive/`
- เรนเดอร์เฟรมเดียว (Single Frame) หรือช่วงเฟรม (Animation: start / end / step)
- Engine: `CYCLES`, `BLENDER_EEVEE_NEXT` / `BLENDER_EEVEE`
- GPU: `OPTIX` (ค่าเริ่มต้น) ถ้าไม่มีให้ใช้ `CUDA` และถ้าไม่มี GPU เลยจะใช้ CPU
- Resolution 25%–200%, Samples override (0 = ค่าในซีน), Output `PNG` / `JPEG` / `OPEN_EXR`
- แสดงความคืบหน้า เฟรมปัจจุบัน เวลาที่ผ่านไป และ log ของ Blender แบบเรียลไทม์
- ปุ่ม Stop / Cancel, ดูภาพเฟรมล่าสุด, Gallery และดาวน์โหลด ZIP เมื่อเสร็จ
- ผลลัพธ์ถูกเขียนลง Drive ทันทีทีละเฟรม ถ้า Session หลุด เฟรมที่เสร็จแล้วจะยังอยู่

### ก่อนใช้งาน: แพ็กพื้นผิว (Textures) เข้าไฟล์ Blender
Colab รันบนเครื่องอื่นจึงมองไม่เห็นไฟล์รูปภาพที่อยู่บนเครื่องคุณ
1. เปิดไฟล์ในโปรแกรม Blender บนเครื่องคุณ
2. ไปที่ **File > External Data > Pack Resources**
3. บันทึกไฟล์ (`Ctrl+S`) แล้วนำไฟล์ `.blend` ที่บันทึกแล้วไปอัปโหลด

### วิธีรันบน Colab
1. อัปโหลด/push โปรเจกต์นี้ขึ้น GitHub แล้วแก้ค่า `REPO_URL` ในเซลล์ `clone-project` ของ `Blender_Colab_Render_UI.ipynb`
2. เปิด `Blender_Colab_Render_UI.ipynb` บน Google Colab
3. ตั้งค่า **Runtime > Change runtime type > Hardware accelerator: GPU**
4. กด **Runtime > Run all** (หรือรันทีละเซลล์)
5. รอจนเห็นลิงก์ `https://xxxx.gradio.live` แล้วเปิดในเบราว์เซอร์ได้จากทุกอุปกรณ์
6. ผลลัพธ์จะอยู่ที่ `Google Drive/MyDrive/Blender_Renders/<ชื่อไฟล์>_<เวลา>/`

ข้อจำกัดของ Colab: Session มีเวลาจำกัด (ประมาณ 12 ชั่วโมง) และ GPU อาจไม่ว่างเสมอ ถ้า Session หลุด ให้รันโน้ตบุ๊กใหม่ แล้วเรนเดอร์ต่อจากเฟรมที่ยังไม่มี

### รันบนเครื่องตัวเอง (ทดสอบ)
```bash
pip install -r requirements.txt
GRADIO_SHARE=0 BLENDER_BIN=/path/to/blender python app.py
```
ต้องมี Blender 4.x สำหรับ Linux และการ์ด NVIDIA ที่รองรับ OptiX/CUDA ถ้าต้องการ GPU

### ทดสอบ
```bash
pytest tests/test_cli_builder.py
```

---

## English

### Features
- Upload a `.blend` file or pick one from `Google Drive/MyDrive/`
- Single-frame or animation-range rendering (start / end / step)
- Engines: `CYCLES`, `BLENDER_EEVEE_NEXT` / `BLENDER_EEVEE`
- GPU: `OPTIX` first, then `CUDA`; falls back to CPU if no GPU is usable
- Resolution scale 25–200%, sample override (0 = scene default), output `PNG`, `JPEG` or `OPEN_EXR`
- Live progress, frame number, elapsed time and Blender log lines
- Stop/Cancel button, latest-frame preview, output gallery and a ZIP of all frames when finished

### Before you render: pack external data
Colab cannot see image files on your computer, so pack them into the scene first:
1. Open the scene in Blender on your computer.
2. Go to **File > External Data > Pack Resources**.
3. Save (`Ctrl+S`) and upload the saved `.blend` file.

### Running on Colab
1. Push this project to GitHub and set `REPO_URL` in the `clone-project` cell of `Blender_Colab_Render_UI.ipynb`.
2. Open the notebook in Google Colab.
3. Set **Runtime > Change runtime type > Hardware accelerator: GPU**.
4. Run all cells. The last cell prints a public `*.gradio.live` link.
5. Output goes to `Google Drive/MyDrive/Blender_Renders/<scene>_<timestamp>/`.

Colab sessions have time limits (about 12 hours) and GPUs are not always available. If the session disconnects, rerun the notebook and render the missing frames again.

### Running locally (for testing)
```bash
pip install -r requirements.txt
GRADIO_SHARE=0 BLENDER_BIN=/path/to/blender python app.py
```

### Tests
```bash
pytest tests/test_cli_builder.py
```

### Project layout
| Path | Purpose |
| :--- | :--- |
| `app.py` | Gradio UI: Single Frame, Animation and Output Gallery tabs |
| `core/installer.py` | Downloads and extracts the Blender Linux `.tar.xz`, validates it, checks the GPU |
| `core/gpu_setup.py` | Runs inside Blender via `-P`; enables OptiX/CUDA devices and applies overrides |
| `core/renderer.py` | Builds the `blender -b ...` argv, streams and parses stdout, supports cancel |
| `Blender_Colab_Render_UI.ipynb` | One-click Colab notebook |
| `tests/test_cli_builder.py` | Tests for command building, path handling, output parsing and cancel |

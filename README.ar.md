# TAIT — Termux AI Training

**أداة خفيفة لتدريب وتشغيل نماذج ذكاء اصطناعي محليًا باستخدام NumPy، ومصممة لـ Termux وأندرويد والأجهزة محدودة ال[...]

[English](README.md) · [العربية](README.ar.md) · [简体中文](README.zh-CN.md)

بدأ TAIT كسكربت واحد لتدريب MiniGPT، وأصبح الآن مشروع Python منظمًا وقابلًا للمساهمة، مع الحفاظ على الفلسفة الخفيفة:[...]

## ✨ الميزات

- تدريب واستدلال MiniGPT باستخدام NumPy
- Byte-level BPE tokenizer
- بيانات JSONL
- مصدر اختياري لـ Hugging Face
- ملفات إعداد TOML مع أولوية خيارات CLI
- معالج إعداد أول تشغيل ومتوافق مع Termux
- `doctor` و`benchmark`
- محادثة طرفية أو محادثة عبر المتصفح محليًا
- Checkpoints وملفات نماذج `.npz`
- بنية مناسبة للمساهمين

## 🚀 التثبيت السريع

```bash
pip install tait
tait setup
```

في Termux يُفضّل:

```bash
pkg install python python-numpy
```

## ⚡ التجربة السريعة

```bash
tait doctor
tait benchmark
tait train --data examples/datasets/demo.jsonl --epochs 3 --output demo.npz
tait chat --model demo.npz
tait chat --web --model demo.npz
```

## 📚 التوثيق

راجع [فهرس التوثيق](docs/README.md) أو الأدلة الموجودة في مجلد `docs/` للتفاصيل.

## 🤝 المساهمة

اقرأ [CONTRIBUTING.md](CONTRIBUTING.md) و[دليل التطوير](docs/development.md).

## 🗺️ خارطة الطريق

يركز v2 على أساس تدريبي ومحادثي خفيف ومنظم. الإصدار v3 المخطط يمكن أن يضيف Agent ووظائف وأدوات دون خلطها مع قلب v2.

## ❤️ الدعم

سيتم نشر طريقة الدعم والتبرع هنا بعد اعتماد الطريقة الرسمية.

## الرخصة

MIT — راجع [LICENSE](LICENSE).

صنع في السودان 🇸🇩

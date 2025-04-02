# Content-Aware Rotation – Modified Version

This is a modified implementation of [Kaiming He et al.'s](http://kaiminghe.com/publications/iccv13car.pdf) **Content-Aware Rotation** (ICCV 2013), originally reproduced by [ForeverPs](https://github.com/ForeverPs/content-aware-rotation).

## 🔧 What's New

This fork includes:
- Code refactoring for readability and modularity
- Add Image preprocess to enhance the result
- Additional test cases or example outputs
- Bug fixes for improved stability across systems

> ✅ Tested on: Windows 10, Linux

---

## 📦 Dependencies

- Python 3.6+
- `pillow==5.1.0`
- `numpy==1.14.5`
- `opencv-python==4.2.0`
- `matplotlib==2.2.2`
- `tensorflow==1.10.0`

---

## 📐 Mathematical Formulation

The original paper introduces an optimization-based method for rotating images content-awarely to correct angles while preserving structural integrity.

All formula derivations and illustrations are retained from the original repository, including:
- Rotation manipulation
- Line, shape, and boundary preservation
- Optimization steps (fix θ, solve for V and vice versa)

📷 See all equations and derivations in the original [ForeverPs repo](https://github.com/ForeverPs/content-aware-rotation).

---

## 🚀 Usage

Clone the repo and run:

```bash
python main.py

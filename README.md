# frb_spectral_search
Measuring scintillation in the spectrum of FRBs with CHIME/FRB Baseband Data 
Maintainers: Eve Schoen, Calvin Leung


# Installation Instructions
Eventually using a more up-to-date build system than `setup.py` would be nice, but the new `pyproject.toml` method does not (yet) support editable installations, which is a non-starter for prototyping pipelines.

```bash
 cd /path/to/install
 git clone https://github.com/eve-schoen/frb-spectral-search.git
 git checkout packaging-cl
 pip install /path/to/install -e #-e makes the installation editable
```

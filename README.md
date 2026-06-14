1. create colmap 3d model
     1. export for ace training `colmap_to_ace.py`
     2. train ace `ace/train_ace.py`
2. create projection `project_scene.py`
3. allign with floor plan   `align_pointmap_to_floorplan.py`
4. localize photo  `ace/ace_loc.py`

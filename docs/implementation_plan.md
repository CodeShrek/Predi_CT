# Phase 1 Re-Architecture Plan

## The Problem
Our scientific audit proved that the current Phase 1 (Atlas-based Registration) is fundamentally flawed for the COCA dataset. 
- **The Data:** COCA consists of Non-Contrast CTs (NCCT). Blood is completely dark and indistinguishable from the surrounding heart muscle and epicardial fat.
- **The Algorithm:** Atlas registration relies on matching pixel intensities (Mutual Information). Because it can't see the blood, it just aligns the overall shape of the heart and drops the arteries into the epicardial fat (missing the true vessel by millimeters).
- **The Consequence:** Running fluid dynamics (Phase 2) on a fake vessel geometry yields scientifically invalid results. 

## Why a "Quick Fix" Won't Work
We cannot simply "force" the algorithm to align the arteries with the bright calcium spots. If we do that, we commit **Circular Logic**: we would be artificially bending the artery to touch the calcium, which would guarantee that Phase 2 finds a correlation, completely invalidating the science.

## The Selected Solution: Population Cohort Approach (Option 3)

Per your request, we will leave the previous `masking.py` and `run_phase2.py` files untouched. We will create entirely new scripts to execute this population-based approach.

### The Strategy
Instead of trying to map calcium and hemodynamics on the exact same patient (which fails because NCCT hides the vessels), we will map them on **two separate populations** and overlay the statistical heatmaps in a standardized reference space.

### Step-by-Step Implementation

1. **Step 1: The ESS Population Heatmap (New File: `generate_ess_atlas.py`)**
   - We will use the 200 `ImageCAS` CCTA scans. These have contrast, so we know exactly where the arteries are.
   - We will run the Phase 2 PINN on a subset of these healthy, high-quality coronary arteries to calculate the true physiological Endothelial Shear Stress (ESS).
   - We will project these ESS values onto a standardized 3D reference heart, creating a statistical heatmap of where shear stress is naturally lowest across a healthy population.

2. **Step 2: The Calcium Population Heatmap (New File: `generate_calcium_atlas.py`)**
   - We will use the `COCA` NCCT scans.
   - We will extract just the calcium locations from all patients and map their 3D coordinates onto the *same* standardized 3D reference heart.
   - This creates a probabilistic heatmap of where calcium most frequently forms in diseased patients.

3. **Step 3: The Statistical Proof (New File: `correlate_population_atlases.py`)**
   - We will calculate the spatial correlation between the low-ESS zones (from Step 1) and the high-calcium zones (from Step 2) on the shared reference heart.
   - A strong correlation proves your hypothesis: the locations where human anatomy naturally creates low shear stress are the exact locations where calcium eventually builds up.

## Open Questions
- To build the ESS atlas quickly, I propose we run the PINN on just **5 representative ImageCAS patients** first to validate the pipeline, before scaling up. Does this sound good?
- We will need a "Standardized Reference Heart" to project both populations onto. I propose using `ImageCAS` patient #1 as the reference template. Do you agree?

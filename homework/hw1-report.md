# ME566 - Homework 1

**AI use:** OpenAI Codex was used to help write and revise the Python code, explain the methods and equations, and organize and edit this report.

The code is in [hw1.ipynb](hw1.ipynb) and [utils](utils/). Environment setup and notebook instructions are in the [README](../README.md). Paths written in code refer to the repository root unless stated otherwise.

Input images

| Question | Exact input under `images/` | Processing |
| --- | --- | --- |
| 1 | `filtered_set_me566_1/` | All readable images, sorted by filename; grayscale; 7 x 10 inner corners. |
| 2 and 3 | `camera_man_w_noise.jpg`, `SaltAndPepperNoise.jpg`, `GaussianNoise.jpg`, `UniformNoise.jpg` | Each JPG loaded separately in grayscale; no added synthetic noise. |
| 4 and 5 | `coins1.jpg`, `coins2.jpg`, `screws.jpeg` | Grayscale; question 4 smooths before thresholding; question 5 operates on the resulting masks. |

Let $I(x,y)$ denote input brightness at column $x$, row $y$. Arrays are indexed as `image[y, x]`. Inputs are 8-bit grayscale, with values from 0 to 255. All comparisons use that same display range. Output arrays retain the input dimensions; saved figures are labeled visualizations, not raw segmentation files. Keep originals for pixel-level evaluation. No matching clean references or manually labeled masks have been established for questions 2-5, so their accuracy metrics below are definitions, not reported measurements.

1. Camera calibration

Implementation: [CalibrateCamera.py](utils/CalibrateCamera.py) and [CalibrationComparison.py](utils/CalibrationComparison.py). Only folder 1 is used. The three groups are subsets of a single recording, so this does not demonstrate three independently acquired data sets if the instructor requires separate recordings.

1. `ImageReader` loads all readable images in lexicographic filename order. `CalibrateCamera` requires a consistent resolution and looks for 70 inner corners using `findChessboardCorners(gray, (7, 10))`.
2. Successful detections are refined with `cornerSubPix`, using `(11, 11)` as the search half-window, `(-1, -1)` for no excluded central region, and termination after at most 30 iterations or a position change below 0.001 pixels. The half-window describes an actual 23 x 23 neighborhood. Images with failed detections are skipped.
3. The board is modeled as a flat grid with unit square spacing. Set the measured square width in `square_size` if physical translations are needed. Intrinsic focal lengths remain in pixels.
4. `default_rng(0).permutation(...)` produces the repeatable index order. `array_split(order1, 3)` creates three disjoint groups. Each group is calibrated using all its usable corners and `CALIB_USE_LU` for the linear solve. The remaining optimization defaults are those of the installed OpenCV version.
5. The first 2, 5, 10, 15 and 25 entries of that same shuffled order provide nested trials. Images in a smaller trial remain in the larger trial. No outliers are removed beyond failed corner detections.

For board indices $a=0,\ldots,6$ and $b=0,\ldots,9$, square width $s$, and image $i$, the board point and its camera coordinates are

$$
\mathbf P_{ab}=(as,bs,0)^T,\qquad
(X_c,Y_c,Z_c)^T=R_i\mathbf P_{ab}+\mathbf t_i.
$$

$R_i$ is a 3 x 3 rotation and $\mathbf t_i$ is translation for that board pose. The returned `rvecs` represent these rotations as Rodrigues rotation vectors, not Euler angles. Normalized image coordinates are $x_n=X_c/Z_c$ and $y_n=Y_c/Z_c$. Define $r^2=x_n^2+y_n^2$ and radial multiplier $L=1+k_1r^2+k_2r^4+k_3r^6$. The five-coefficient distortion model is

$$
x_d=x_nL+2p_1x_ny_n+p_2(r^2+2x_n^2),
$$

$$
y_d=y_nL+p_1(r^2+2y_n^2)+2p_2x_ny_n.
$$

$k_1,k_2,k_3$ describe radial distortion; $p_1,p_2$ describe tangential distortion. Pixel predictions and the intrinsic camera matrix are

$$
\hat u=f_xx_d+c_x,\qquad \hat v=f_yy_d+c_y,\qquad
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
$$

$f_x,f_y$ are focal lengths in pixels, and $(c_x,c_y)$ is the principal point. The solver adjusts the intrinsics, distortion and per-image poses to reduce squared reprojection error. If image $i$ has $N_i$ observed corners $\mathbf q_{ij}=(u_{ij},v_{ij})$, then

$$
E=\sum_i\sum_{j=1}^{N_i}\|\mathbf q_{ij}-\hat{\mathbf q}_{ij}\|_2^2,
\qquad \mathrm{RMS}=\sqrt{\frac{E}{\sum_i N_i}}.
$$

The Euclidean norm includes both pixel-coordinate errors. RMS has pixel units and measures fit to the calibration images, not accuracy on unseen images. The model and API are documented in [OpenCV calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

The initial calibration found corners in 813 of 816 images, giving three groups of 271. The tables below are from that run. The code has since been moved into utility scripts and has not been rerun to verify those changes.

| Group | RMS (pixels) | fx | fy | cx | cy |
| --- | --- | --- | --- | --- | --- |
| 1A | 0.7499 | 1765.75 | 1762.70 | 738.14 | 602.68 |
| 1B | 1.1473 | 1764.15 | 1760.83 | 756.65 | 617.31 |
| 1C | 0.9627 | 1748.98 | 1746.22 | 747.97 | 616.39 |

| Images | RMS (pixels) | fx | fy | cx | cy |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.1659 | 1642.64 | 1638.80 | 705.56 | 686.55 |
| 5 | 0.1963 | 1665.02 | 1660.50 | 682.73 | 770.00 |
| 10 | 0.1892 | 1737.37 | 1734.49 | 704.90 | 625.73 |
| 15 | 0.3045 | 1798.85 | 1795.29 | 720.83 | 604.62 |
| 25 | 0.8866 | 1792.43 | 1789.45 | 715.47 | 591.92 |

The large groups agree within approximately 1% on focal length, consistent with using the same recording. The two-image result has the smallest RMS but a substantially smaller focal length and a shifted principal point. A small fitting error does not necessarily mean the camera parameters are accurate. The 15- and 25-image focal lengths are closer to each other, but distortion still varies; more diverse poses can help, whereas similar video frames add less information. The program also prints the distortion coefficients. Relative differences against a value $f_{\mathrm{ref}}$ are calculated as $100|f-f_{\mathrm{ref}}|/|f_{\mathrm{ref}}|$ percent.

2. Gaussian and median filtering

The question 2 cell loads each noisy JPG with `ImageReader` and applies a 3x3 Gaussian, 5x5 Gaussian and 3x3 median filter. Each result is displayed beside the original. The filenames and filter settings are kept in the notebook.

Questions 1, 4 and 5 call utility classes. Questions 2 and 3 keep filtering in the notebook and use the shared plotting helper:

| Script in `homework/utils` | Main call | Purpose |
| --- | --- | --- |
| `CalibrationComparison.py` | `CalibrationComparison().compare()` | Runs question 1 using `CalibrateCamera` and returns the calibration results. |
| Notebook questions 2 and 3 | OpenCV filter calls and `rank_filter(...)` | Reads the JPGs, filters them and displays comparisons inline. |
| `ThresholdImages.py` | `ThresholdImages().compare()` | Runs question 4 and returns the original images and masks. |
| `Morphology.py` | `Morphology(segmentation).compare()` | Runs question 5 using the masks returned by question 4. |
| `ShowResults.py` | `show_results(images, titles, filename)` | Displays and saves the comparison figures for questions 2-5. |

`__init__` stores the image folder or input masks on the object as `self.image_dir` or `self.segmentation`. Calling a method then uses that stored input. The default image paths are relative to the scripts, and you can pass a different folder when creating a thresholding or calibration-comparison object. Filter settings are in the notebook; threshold and morphology settings are in their utility scripts.

Functions

- `ImageReader(path, grayscale=True).images[0]` reads one image as a grayscale array. Each number is a pixel brightness from 0 (black) to 255 (white). `[0]` selects the first image from the reader's list.
- `cv2.GaussianBlur(image, (3, 3), 0)` replaces each pixel with a weighted average of its neighbors. Nearby pixels have more weight than distant pixels. `(3, 3)` sets the neighborhood size, and `0` lets OpenCV choose the Gaussian spread, sigma, from that size. The `(5, 5)` call works the same way over a larger neighborhood. Each filter starts from the original image.
- `cv2.medianBlur(image, 3)` sorts the nine brightness values in a 3x3 neighborhood and uses the middle value. For example, `[0, 98, 99, 100, 100, 101, 102, 103, 255]` gives a median of 100. An isolated black or white pixel has less influence than it would on an average.
- `plt.subplots(1, 4)` makes four panels for the original and filtered images. `zip(...)` pairs each panel with its image and title. `imshow(..., cmap="gray", vmin=0, vmax=255)` uses the same brightness scale in every panel so the comparison is fair.
- `fig.savefig(...)` saves each comparison for this writeup. `plt.show()` displays it in the notebook, and `plt.close(fig)` releases the figure afterward.

Settings

A 3x3 Gaussian kernel provides mild smoothing and usually preserves more small details. A 5x5 kernel uses more neighbors and, with automatic sigma, also has a wider Gaussian spread. It usually removes more visible noise but blurs edges more. This compares two practical smoothing settings; both kernel size and the automatic spread change. A 3x3 median filter is a small starting window for removing isolated spikes while limiting detail loss. It can still remove thin lines or small features.

Expected behavior

The following comparison describes expected behavior. Observations from the output images still need to be added.

| Image | What to compare | Expected useful filter |
| --- | --- | --- |
| camera_man_w_noise | Check noise in smooth areas and sharpness around the person. The filename alone does not identify the noise distribution. | Median for isolated black/white dots; Gaussian for distributed grain. |
| SaltAndPepperNoise | Check whether bright/dark dots disappear or become blurry spots. | Median 3x3 should handle isolated impulses better than Gaussian averaging. |
| GaussianNoise | Compare remaining grain against softened edges. | Gaussian 3x3 for detail retention; Gaussian 5x5 for stronger smoothing. |
| UniformNoise | Compare fluctuations in smooth regions and loss of texture. | Gaussian smoothing can reduce independent additive uniform noise too; choose the size by the noise/detail tradeoff. |

There is no single best filter for every image. Averaging helps reduce random fluctuations, but an extreme impulse can affect neighboring output pixels. The median is less sensitive to isolated extremes. Stronger smoothing can make an image look cleaner while also removing useful information.

Quality measurement

For these supplied noisy inputs, the comparison uses visual assessment of residual noise and preservation of edges. This is subjective, not a numerical accuracy metric. A numerical choice would be PSNR against a matching clean reference image: first compute MSE, the mean squared pixel difference, then `PSNR = 10 * log10(255**2 / MSE)` for 8-bit images. Lower MSE and higher PSNR indicate closer agreement with that reference. No matching clean reference has been identified for these four images, so the code does not report PSNR. Comparing the filtered image against the noisy input would measure how much it changed, not how well it recovered the clean image.

Original and filtered images

The question 2 cell saves four figures with these panels: original, Gaussian 3x3, Gaussian 5x5 and median 3x3. The image links below resolve once the figures are saved.

![Camera man comparison](filter_results/camera_man_w_noise_comparison.png)

![Salt and pepper comparison](filter_results/SaltAndPepperNoise_comparison.png)

![Gaussian noise comparison](filter_results/GaussianNoise_comparison.png)

![Uniform noise comparison](filter_results/UniformNoise_comparison.png)

3. Median and n-rank filtering

The question 3 cell compares median and lower-quartile rank filters at both window sizes. It uses `ShowResults.py` to display and save the images.

Functions

- `cv2.medianBlur(image, size)` selects the middle neighborhood value. For nine values this is zero-based rank 4; for 25 values it is rank 12. A median is therefore a particular rank filter.
- `rank_filter(image, size, rank)` selects a different position in that ordered neighborhood. Here, ranks 2 and 6 select approximately the lower quartile for 3x3 and 5x5 windows. This keeps the relative position comparable across sizes. These ranks were chosen because the assignment does not specify n.
- `np.pad(..., mode="edge")` repeats border pixels so the output stays the same size as the input. `sliding_window_view` gathers a neighborhood at each pixel. `reshape` arranges its 9 or 25 values along the last axis. `np.partition(..., rank, axis=-1)` places the selected value where it would occur in a sorted list without fully sorting the list. `[..., rank]` takes that value for every pixel.
- `show_results(images, titles, filename)` is just the plotting loop from question 2 moved into a reusable function. It displays every supplied image and saves the whole labeled comparison as a PNG.

For example, the ordered neighborhood `[0, 98, 99, 100, 100, 101, 102, 103, 255]` gives median 100 and rank-2 value 99. Rank 0 would be a minimum filter; rank 8 would be a maximum filter for this 3x3 window.

Comparison

The median is a useful starting point for salt-and-pepper noise because it rejects isolated extremes on either side. A 5x5 median may remove more impulses but also erase small features. The lower-quartile rank filter favors darker values: it can suppress bright impulses but expand dark ones and darken the result. It is not a general improvement over the median. For Gaussian or uniform grain, compare both against the Gaussian results from question 2, paying attention to remaining grain and lost texture. For the camera-man image, identify whether the visible noise is isolated dots or distributed grain before choosing a filter. Use the same visual criteria and clean-reference PSNR limitation discussed in question 2.

![Camera man rank comparison](filter_results/camera_man_w_noise_rank.png)

![Salt and pepper rank comparison](filter_results/SaltAndPepperNoise_rank.png)

![Gaussian rank comparison](filter_results/GaussianNoise_rank.png)

![Uniform rank comparison](filter_results/UniformNoise_rank.png)

4. Simple and adaptive thresholding

Functions

`cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY_INV)` uses one cutoff everywhere. With inversion, pixels at or below 127 become white (255); brighter pixels become black (0). It returns the cutoff and mask; `_` discards the cutoff. White represents the intended dark foreground.

`cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)` computes a local Gaussian-weighted threshold in a 31x31 neighborhood and subtracts 5. The odd block size and C are adjustable starting settings. With inversion, increasing C generally reduces white foreground. Both methods receive the same 3x3 Gaussian-smoothed image. These follow the [OpenCV thresholding tutorial](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).

Comparison

A global cutoff is easy to interpret, but shadows and highlights may defeat one value. Adaptive thresholding follows local lighting but may also select unwanted texture. Neither method guarantees filled object silhouettes.

| Image | What is visible in the input | What to check after running |
| --- | --- | --- |
| coins1 | Coin brightness varies against a gray background. | Does 127 miss light coins or select shadowed background? Does adaptive retain boundaries but leave holes? |
| coins2 | Many coins lie close together on a light background, with bright details on their faces. | Does either method keep separate, filled coins without losing reflective regions? |
| screws | Reflective screw heads and threads lie on woven fabric. | Does adaptive select the fabric weave? Does global lose bright metal regions or include shadows? |

These checks are based on the original photographs; the output masks still need to be assessed. No manually labeled masks are available for a numerical accuracy comparison. With labels, intersection-over-union (IoU) would measure overlap between the predicted and true foreground.

![Coins1 threshold comparison](filter_results/coins1_threshold.png)

![Coins2 threshold comparison](filter_results/coins2_threshold.png)

![Screws threshold comparison](filter_results/screws_threshold.png)

5. Morphological operations

The prompt refers to segmentation in problem 3, but problem 3 is filtering. This cell uses the binary masks from question 4. Each threshold method is compared with both rectangular and elliptical masks of size 3x3 and 5x5.

Functions

`cv2.getStructuringElement(shape, (size, size))` builds the neighborhood mask, also called a structuring element. Its active positions control which neighbors participate.

- `cv2.erode(mask, kernel)` shrinks white regions and can remove small white specks.
- `cv2.dilate(mask, kernel)` expands white regions and can bridge small gaps.
- `cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)` erodes then dilates to remove small foreground features.
- `cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)` dilates then erodes to fill small dark holes and gaps.

Open-then-close removes specks before filling gaps; close-then-open fills gaps before removing specks. Order matters because the first operation changes the input to the second. One iteration keeps the initial comparison conservative. See the [OpenCV morphology tutorial](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).

Choosing the mask and operation order

The best combination has not yet been determined from the outputs. For separated round coins, an elliptical mask is a sensible starting shape. Try 3x3 opening followed by closing when isolated white specks and small dark holes are the main errors. For close coins, check carefully that closing does not join neighbors. For screws, start small to preserve threads and narrow parts. If holes dominate, try closing first; if fabric speckles dominate, try opening first. A 5x5 mask has a larger effect but can erase useful detail or merge nearby objects. Rectangular masks include more corner neighbors than elliptical masks of the same dimensions, so compare their boundary changes too.

The comparison favors small masks that remove noise while keeping objects separate. Morphology cannot reliably recover an object that thresholding mostly missed, remove all fabric texture without affecting threads, or fill holes much larger than the mask. Poor starting masks may require different thresholds before morphology can help.

Morphology figures

The notebook displays and saves 24 labeled comparison figures: three images x two threshold methods x two shapes x two sizes. Each figure contains the original, threshold mask, erosion, dilation, opening, closing, open-then-close and close-then-open outputs. Filenames follow `image_method_shape_size_morphology.png`. All panels are included in the saved figures in [filter_results](filter_results/).

| Image / threshold | Rectangle 3x3 | Rectangle 5x5 | Ellipse 3x3 | Ellipse 5x5 |
| --- | --- | --- | --- | --- |
| coins1 / global | ![coins1 global rectangle 3](filter_results/coins1_global_rectangle_3_morphology.png) | ![coins1 global rectangle 5](filter_results/coins1_global_rectangle_5_morphology.png) | ![coins1 global ellipse 3](filter_results/coins1_global_ellipse_3_morphology.png) | ![coins1 global ellipse 5](filter_results/coins1_global_ellipse_5_morphology.png) |
| coins1 / adaptive | ![coins1 adaptive rectangle 3](filter_results/coins1_adaptive_rectangle_3_morphology.png) | ![coins1 adaptive rectangle 5](filter_results/coins1_adaptive_rectangle_5_morphology.png) | ![coins1 adaptive ellipse 3](filter_results/coins1_adaptive_ellipse_3_morphology.png) | ![coins1 adaptive ellipse 5](filter_results/coins1_adaptive_ellipse_5_morphology.png) |
| coins2 / global | ![coins2 global rectangle 3](filter_results/coins2_global_rectangle_3_morphology.png) | ![coins2 global rectangle 5](filter_results/coins2_global_rectangle_5_morphology.png) | ![coins2 global ellipse 3](filter_results/coins2_global_ellipse_3_morphology.png) | ![coins2 global ellipse 5](filter_results/coins2_global_ellipse_5_morphology.png) |
| coins2 / adaptive | ![coins2 adaptive rectangle 3](filter_results/coins2_adaptive_rectangle_3_morphology.png) | ![coins2 adaptive rectangle 5](filter_results/coins2_adaptive_rectangle_5_morphology.png) | ![coins2 adaptive ellipse 3](filter_results/coins2_adaptive_ellipse_3_morphology.png) | ![coins2 adaptive ellipse 5](filter_results/coins2_adaptive_ellipse_5_morphology.png) |
| screws / global | ![screws global rectangle 3](filter_results/screws_global_rectangle_3_morphology.png) | ![screws global rectangle 5](filter_results/screws_global_rectangle_5_morphology.png) | ![screws global ellipse 3](filter_results/screws_global_ellipse_3_morphology.png) | ![screws global ellipse 5](filter_results/screws_global_ellipse_5_morphology.png) |
| screws / adaptive | ![screws adaptive rectangle 3](filter_results/screws_adaptive_rectangle_3_morphology.png) | ![screws adaptive rectangle 5](filter_results/screws_adaptive_rectangle_5_morphology.png) | ![screws adaptive ellipse 3](filter_results/screws_adaptive_ellipse_3_morphology.png) | ![screws adaptive ellipse 5](filter_results/screws_adaptive_ellipse_5_morphology.png) |



Equations and processing steps

Question 2: Gaussian filtering

For an odd window width $m$, let $h=(m-1)/2$ and offsets $a,b\in\{-h,\ldots,h\}$. A normalized sampled Gaussian has weights

$$
G_\sigma(a,b)=\frac{\exp[-(a^2+b^2)/(2\sigma^2)]}{\sum_{c=-h}^{h}\sum_{d=-h}^{h}\exp[-(c^2+d^2)/(2\sigma^2)]},
\qquad J(x,y)=\sum_{a=-h}^{h}\sum_{b=-h}^{h}G_\sigma(a,b)I(x+a,y+b).
$$

$\sigma$ is the spatial spread in pixels, $G_\sigma$ is a weight, and $J$ is the smoothed image. Normalization makes the weights sum to one, preserving a constant region's brightness. This is the general mathematical model; the code passes sigma zero to let OpenCV select its discrete kernel. For exact reproduction use the supplied `GaussianBlur` calls, rather than choosing a separate sigma for this formula. The implementation can use predefined small kernels; `cv2.getGaussianKernel(m, 0)` reveals its one-dimensional coefficients, and their outer product gives the two-dimensional kernel. Output is rounded to 8-bit brightness.

For each of the four noisy JPGs: load grayscale, independently compute Gaussian 3x3, Gaussian 5x5 and median 3x3 from that original, then display them in that order after the original. Do not feed the 3x3 result into the 5x5 filter. Gaussian smoothing retains OpenCV's default reflected boundary handling (`BORDER_REFLECT_101`); the median uses replicated boundaries. These border rules can produce different edge pixels even if the interior looks similar. See [OpenCV filtering functions](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html).

Question 3: median and n-rank filtering

Sort the $N=m^2$ neighborhood values conceptually as $z_{(0)}\le\cdots\le z_{(N-1)}$. Then

$$
J_r(x,y)=z_{(r)},\qquad
J_{\mathrm{median}}(x,y)=z_{((N-1)/2)}.
$$

$r$ is a zero-based rank, not a brightness or a window size. Our nonmedian choice is $r=\lfloor(N-1)/4\rfloor$: rank 2 for $m=3$ and rank 6 for $m=5$. The median instead uses ranks 4 and 12. The floor symbol means round down to the nearest integer. A minimum selects rank 0 and a maximum selects rank $N-1$.

For each noisy image, independently compute median 3x3, median 5x5, rank-2 3x3 and rank-6 5x5 from the original. Display the original followed by those four outputs. The custom rank function repeats the nearest edge value when its neighborhood extends beyond the image, matching the median's replicated padding. Partitioning computes the chosen order statistic; a full sort is unnecessary. A lower rank favors darker values, explaining why it suppresses bright impulses but can enlarge dark ones.

MSE and PSNR

For a clean reference $F$ and filtered result $J$ with height $H$ and width $W$,

$$
\mathrm{MSE}=\frac{1}{HW}\sum_{y=0}^{H-1}\sum_{x=0}^{W-1}[J(x,y)-F(x,y)]^2,
\qquad
\mathrm{PSNR}=10\log_{10}\left(\frac{255^2}{\mathrm{MSE}}\right).
$$

MSE is mean squared brightness error; PSNR is measured in decibels. Identical images give MSE zero and infinite PSNR. Convert arrays to floating point before subtracting to avoid unsigned 8-bit wraparound. The reference must depict the same scene with the same alignment, dimensions and brightness scale. The available code does not compute these metrics because no appropriate clean references have been identified. For visual comparison, inspect the same smooth region and the same edge in every panel, and report the location, remaining noise and detail loss. This gives a repeatable inspection procedure, though it remains subjective.

Question 4: thresholding equations and sequence

Let $S$ be the grayscale image after the 3x3 Gaussian blur. The inverted global mask is

$$
M_g(x,y)=\begin{cases}255&S(x,y)\le127,\\0&S(x,y)>127.\end{cases}
$$

For adaptive Gaussian thresholding, use local weights $w(a,b)$ that sum to one across the 31x31 block:

$$
\mu_w(x,y)=\sum_{a=-15}^{15}\sum_{b=-15}^{15}w(a,b)S(x+a,y+b),\qquad T(x,y)=\mu_w(x,y)-5,
$$

$$
M_a(x,y)=\begin{cases}255&S(x,y)\le T(x,y),\\0&S(x,y)>T(x,y).\end{cases}
$$

$\mu_w$ is local weighted brightness, $T$ is the local cutoff and 5 is the constant C in brightness levels. OpenCV computes the local mean using its Gaussian implementation and 8-bit rounding; use the API for exact threshold decisions near the cutoff. Adaptive thresholding uses replicated image boundaries. A 31-pixel block is a starting compromise between very local texture and larger lighting changes, not a fitted optimum.

For each of the three images: load grayscale, blur once, compute both masks from that same blurred image, save `(original, global, adaptive)` in `segmentation[name]`, and display original, smoothed image and both masks. Do not invert again before morphology: white is already the intended foreground. Edit `127`, `31` and `5` in `ThresholdImages.py` if testing alternatives, and record every changed value. The local and fixed-cutoff procedures follow the [OpenCV thresholding tutorial](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).

For predicted foreground set $P$ and manually labeled true foreground set $Q$,

$$
\mathrm{IoU}=\frac{|P\cap Q|}{|P\cup Q|}.
$$

Vertical bars count pixels, the intersection counts pixels selected correctly, and the union counts pixels selected in either mask. IoU ranges from zero to one when the union is nonempty. If both sets are empty, report an explicit convention instead of dividing by zero. No IoU has been computed without labels.

Question 5: morphology equations and sequence

Let $M(x,y)$ be a 0/255 mask and $B$ the set of active offsets in a centered, symmetric structuring element. Erosion and dilation are neighborhood minimum and maximum:

$$
E_B(M)(x,y)=\min_{(a,b)\in B}M(x+a,y+b),\qquad
D_B(M)(x,y)=\max_{(a,b)\in B}M(x+a,y+b).
$$

The minimum stays white only when every active neighbor is white; the maximum becomes white if any active neighbor is white. Opening and closing are compositions:

$$
O_B(M)=D_B(E_B(M)),\qquad C_B(M)=E_B(D_B(M)).
$$

The tested two-stage sequences are

$$
\text{open then close}=C_B(O_B(M)),\qquad
\text{close then open}=O_B(C_B(M)).
$$

Read the innermost operation first. The same $B$ is used at every stage of a sequence. All operations use one iteration, centered anchors, and OpenCV's default morphology border value (neutral for the relevant minimum or maximum). A 3x3 rectangle activates all nine positions, while a 3x3 ellipse is a cross. The 5x5 ellipse generated by OpenCV is a discrete approximation, not a filled 5x5 square. The exact active pixels come from `getStructuringElement`, as shown in the code.

For each image and each original threshold mask, try each shape and size independently. Compute erosion, dilation, opening and closing directly from that mask. Compute open-then-close from the opening and close-then-open from the closing. Never carry the previous shape's or size's output into a new trial. Display all six processed masks together with the original photograph and starting mask. This isolates shape, size and order effects.

Remaining results

The figures and final image-by-image findings for questions 2-5 still need to be checked after running the notebook. Each comparison should identify the selected settings, the visible improvement and any lost detail. For morphology, this includes whether coins remain separate and whether screw threads are preserved.

Input JPGs and output PNGs are excluded by the current `.gitignore`. They need to be available with the submission for the figure links and reproduction steps to work.


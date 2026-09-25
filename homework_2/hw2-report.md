# Homework 2: local features on camera images

The executable figures and tables are in [hw2.ipynb](hw2.ipynb). The Python implementations are in [`utils/`](utils/), with experiment orchestration in [`FeatureComparison.py`](utils/FeatureComparison.py). Install [`requirements.txt`](../requirements.txt) and run the notebook from the repository root or `homework_2`. The first cell prints package versions for the exact run. OpenCV is the main image-processing library; the DoG implementation uses `skimage.feature.blob_dog` for scale-space peaks.

For Linux setup and notebook instructions, see the [Homework 2 README](README.md).

## 1. Detector comparison

### Images and procedure

The image folders are `images_hw2/images/hw2_translation_images` (15 frames, `frame_130.jpg` through `frame_200.jpg`), `hw2_scale_images` (21 frames, `frame_100.jpg` through `frame_200.jpg`), and `hw2_rot_images` (21 frames, `frame_100.jpg` through `frame_200.jpg`). Every file is 1456 × 1088 pixels. [`read_images_from_folder`](utils/ImageReader.py) sorts frames by the number in the filename and loads each with `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`. The motion is physical camera motion. The code never digitally rotates, scales, resizes, or warps a detector input. The three overlay figures show the first, middle, and last recorded view of each motion. The numeric comparison uses the first frame as reference, every 5 images, and the final image. Thus rotation and scale compare frames 105, 130, 155, 180, and 200 with frame 100; translation compares frames 135, 160, 185, and 200 with frame 130. Frame numbers are ordering labels, not measured angles or scale factors.

Each detector returns image-coordinate points `(x,y)`. DoG peaks are blob centers; the other methods return corners, so their counts should not be treated as identical feature types. For image $i$ and detector $d$, the **feature count** is $N_{i,d}=|P_{i,d}|$. **Detection time** is $t_{i,d}=1000(t_{\mathrm{end}}-t_{\mathrm{start}})$ milliseconds, measured with `time.perf_counter` around detection alone. Loading, plotting, homography fitting, and repeatability scoring are outside this timer.

To compare the same scene locations, a SIFT script estimates a homography $H$ between the reference and later photograph. OpenCV SIFT detects at most 3000 registration features per image. Brute-force L2 matching finds two neighbors per descriptor; a candidate is retained if $d_1/d_2<0.7$. `cv2.findHomography(..., cv2.RANSAC, 3.0)` estimates $H$ with a 3-pixel reprojection threshold. RANSAC repeatedly fits a homography to small match subsets and rejects matches that disagree with it; fewer than 10 inliers make registration invalid. `cv2.perspectiveTransform` maps **point coordinates only**. For a reference point $p$, let $p'=\pi(H[p_x,p_y,1]^T)$, where $\pi([u,v,w]^T)=(u/w,v/w)$. Only points whose mapped locations lie inside the later image count as eligible. The implemented repeatability is

$$
R=\frac{\#\{p\in P_{\mathrm{ref}}^{\mathrm{visible}}:\min_{q\in P_{\mathrm{later}}}\lVert p'-q\rVert_2\le 5\,\mathrm{pixels}\}}{|P_{\mathrm{ref}}^{\mathrm{visible}}|}.
$$

This is a one-direction location score, not a unique one-to-one match count. The reference frame is assigned $R=1$. A missing homography or no eligible points produces `NaN`. Registration uses SIFT rather than the evaluated detector, but it can still be biased by SIFT failures. Parallax, occlusion, and nonplanar scene content also affect $R$; a dense detector has more opportunities to put a point within 5 pixels. Read repeatability alongside feature count.

### Detector equations and parameters

| Method | Implemented test and settings | Meaning |
|---|---|---|
| DoG | $L(x,y,\sigma)=G_\sigma*I$; $D(x,y,\sigma)=L(x,y,\sigma)-L(x,y,k\sigma)$. `blob_dog` on grayscale float image in `[0,1]`, `min_sigma=1`, `max_sigma=12`, `threshold=0.03`; default `sigma_ratio=1.6`, `overlap=0.5`. | Gaussian scale-space extrema identify blobs of several sizes. `threshold` rejects weak peaks. The output sigma is multiplied by $\sqrt2$ for display radius; the experiment counts only centers. |
| Harris | $M=\sum_W \begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}$, $R_H=\det(M)-0.04\,\mathrm{tr}(M)^2$. `cornerHarris(blockSize=2, ksize=3, k=0.04)` on float grayscale. Keep local maxima above `0.01 * max(response)`. | `blockSize` is the neighborhood for the gradient matrix; `ksize` is the Sobel aperture; $k$ controls the edge penalty. Dilating the response with a default 3×3 neighborhood identifies local maxima. |
| Shi–Tomasi | $R_{ST}=\min(\lambda_1,\lambda_2)$, where $\lambda_1,\lambda_2$ are eigenvalues of $M$. `goodFeaturesToTrack(maxCorners=2000, qualityLevel=0.01, minDistance=5, blockSize=3)` with default `useHarrisDetector=False`. | Retains corners above 1% of the strongest quality, at least 5 pixels apart, up to 2000. The 2000 cap makes counts partly parameter-dependent. |
| FAST | A pixel is a corner when 9 contiguous pixels on its 16-pixel circle are all brighter than center + 20 or all darker than center − 20. `FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)` uses default `TYPE_9_16`. | The 20-gray-level test rejects small intensity changes. Nonmaximum suppression reduces nearby duplicate responses. |

These equations describe the score or corner rule; OpenCV and scikit-image perform the local filtering and selection. Sources: [OpenCV corner functions](https://docs.opencv.org/4.11.0/dd/d1a/group__imgproc__feature.html), [OpenCV FAST](https://docs.opencv.org/4.13.0/df/d74/classcv_1_1FastFeatureDetector.html), [scikit-image DoG](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog).

### Observed results

The saved baseline notebook run gave these **means across the reference and sampled frames**. Its reference repeatability of 1 is included in each mean, so the averages are higher than they would be for changed views alone. Values may vary on another computer, especially timings.

| Motion | Detector | Mean features | Mean detection time (ms) | Mean repeatability |
|---|---|---:|---:|---:|
| Rotation | DoG | 1298.7 | 877.4 | 0.585 |
| Rotation | Harris | 199.5 | 28.2 | 0.412 |
| Rotation | Shi–Tomasi | 914.8 | 15.1 | 0.567 |
| Rotation | FAST | 591.5 | 1.6 | 0.601 |
| Scale | DoG | 1232.8 | 693.5 | 0.465 |
| Scale | Harris | 391.8 | 30.0 | 0.399 |
| Scale | Shi–Tomasi | 1060.0 | 18.8 | 0.571 |
| Scale | FAST | 572.2 | 1.3 | 0.444 |
| Translation | DoG | 1657.0 | 763.1 | 0.471 |
| Translation | Harris | 228.2 | 31.5 | 0.475 |
| Translation | Shi–Tomasi | 1097.0 | 15.0 | 0.526 |
| Translation | FAST | 1051.4 | 1.6 | 0.522 |

FAST was the quickest detector in all three sequences. DoG detected the most points and was far slower because it searches several Gaussian scales. Shi–Tomasi had the highest repeatability for scale and translation, while FAST was highest for rotation in this run. Harris found the fewest points in each sequence and had the lowest rotation and scale repeatability. These are results for the specified thresholds and scene, not general rankings. The notebook's frame-by-frame plots show how each quantity changes as the camera moves; physical angle, displacement, and scale ratio were not recorded.

## 2. Spatially distributed subset: ANMS

The implemented method is **adaptive** nonmaximum suppression (ANMS) from Brown, Szeliski, and Winder, *Multi-Image Matching using Multi-Scale Oriented Patches*, CVPR 2005, Section 3 ([paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf)). It starts from Harris local maxima with positive response $s_i$. For each candidate location $x_i$, define its suppression radius by

$$
r_i^2=\min_{j:s_j>s_i/0.9}\lVert x_i-x_j\rVert_2^2.
$$

with $r_i=\infty$ if no substantially stronger point exists. The code stores squared distances, which produce the same ordering as the paper's distances. The paper titled *Asynchronous Non-Maximum Suppression in Event Stream* addresses event-camera data and is a different method. The 0.9 factor requires the competing point to be meaningfully stronger. Keep the 100 largest radii for the demonstration. Thus isolated strong corners survive even when many strongest-response corners cluster in one region. The notebook compares ANMS with choosing the 100 largest Harris responses. It reports how many of a 4×4 image grid contain at least one selected point; coverage is $C=\#\{\text{occupied cells}\}/16$. On the reference translation frame, the strongest 100 points occupied 9/16 cells and ANMS occupied 11/16 cells. This is a simple distribution check, not proof that all image regions contain detectable corners. The reference translation view has 168 Harris candidates, so a 300-point cap would leave that set unchanged; the demonstration uses 100 points. The descriptor experiment below uses the same Harris candidate rule with ANMS capped at 300 points per image, so images with fewer than 300 candidates are unchanged.

## 3. SIFT, BRIEF, and ORB descriptor matching

The detection scheme is fixed to Harris response peaks followed by ANMS (up to 300 points) for **all three descriptors**. Each supplied point becomes `cv2.KeyPoint(x, y, size=16)`. Each descriptor's `compute(image, keypoints)` may discard border points, so the final descriptor counts can differ. The code does not assign a measured orientation to the supplied keypoints; rotation performance here is for this particular fixed-keypoint setup, not the full detector-plus-descriptor pipeline of SIFT or ORB.

| Descriptor | Implemented setup | Comparison distance |
|---|---|---|
| SIFT | `cv2.SIFT_create()` with OpenCV defaults; computes a 128-component gradient-histogram vector at the supplied keypoints. | $d(a,b)=\sqrt{\sum_k(a_k-b_k)^2}$ (`NORM_L2`). |
| BRIEF | `cv2.xfeatures2d.BriefDescriptorExtractor_create()` defaults: 32 bytes (256 binary tests), `use_orientation=False`. Each bit compares intensities at two patch locations, $b_k=[I(u_k)<I(v_k)]$. | Hamming count $d(a,b)=\sum_k[a_k\ne b_k]$ (`NORM_HAMMING`). |
| ORB | `cv2.ORB_create()` defaults, including `WTA_K=2`; computes a binary oriented-BRIEF-style descriptor at supplied points. | Hamming count (`NORM_HAMMING`). |

Sources: [OpenCV SIFT](https://docs.opencv.org/4.12.0/d7/d60/classcv_1_1SIFT.html), [OpenCV BRIEF](https://docs.opencv.org/4.13.0/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html), [OpenCV ORB](https://docs.opencv.org/4.5.4/db/d95/classcv_1_1ORB.html), [OpenCV BFMatcher](https://docs.opencv.org/4.12.0/d3/da1/classcv_1_1BFMatcher.html).

For each sampled image pair, `cv2.BFMatcher(norm).knnMatch(..., k=2)` finds the nearest and second-nearest target descriptor for every reference descriptor. The match score is $\rho=d_1/\max(d_2,10^{-12})$; accept the nearest match when $\rho<0.75$. The SIFT-derived homography from Section 1 provides an **approximate geometric label**: a candidate nearest-neighbor match is labelled correct if its target lies within 5 pixels of the projected reference keypoint. Failed registrations are omitted. The matching timer includes Harris+ANMS detection, descriptor extraction for both images, nearest-neighbor search, and geometric scoring; it excludes file loading and the separate homography estimation. Thus it measures this complete per-pair matching pipeline, not descriptor extraction alone. The SIFT-based geometric labels can favor SIFT and are not manually verified ground truth.

The notebook now displays one 128-value SIFT vector as an 8 × 16 intensity map and one 256-bit vector each for BRIEF and ORB as 16 × 16 binary maps. The maps are reshaped for display; neighboring cells in these displays do not imply neighboring image locations. It also overlays the best accepted correspondences on one sampled image pair (green for within 5 pixels of the homography projection, red otherwise). These figures make the descriptor representation and matching errors visible alongside the numerical results. The confusion matrix has rows *geometrically wrong, geometrically right* and columns *rejected, accepted*. With those axes, the cells are [TN, FP; FN, TP]. Accepted-match precision is $TP/(TP+FP)$. The saved baseline run printed the following values. Rerun the notebook after changing settings to update them:

| Descriptor | TN | FP | FN | TP | Mean pair time (ms) | Accepted precision |
|---|---:|---:|---:|---:|---:|---:|
| SIFT | 2857 | 342 | 146 | 327 | 101.74 | 0.489 |
| BRIEF | 2460 | 505 | 99 | 348 | 68.55 | 0.408 |
| ORB | 2413 | 551 | 126 | 297 | 76.17 | 0.350 |

On these camera pairs, SIFT yielded the highest accepted-match precision and the slowest per-pair time; BRIEF was fastest. SIFT's advantage should be read with the SIFT-registration bias and fixed keypoint orientation in mind. The matrices contain one best-match candidate per reference descriptor and aggregate across all sampled motions, not a balanced classification dataset.

## 4. SIFT ratio-test ROC

SIFT is the chosen matching method. Vary its acceptance threshold $\tau$ over all observed ratios, accepting when $\rho<\tau$. At each threshold, $TPR=TP/(TP+FN)$ and $FPR=FP/(FP+TN)$. The ROC plots TPR against FPR; area under it summarizes separation over thresholds. The saved baseline plot has **AUC = 0.867** using the approximate geometric labels above. A ratio threshold of 0.75 is the reported operating point, not a universal optimum. Smaller thresholds reject ambiguous matches and tend to lower FPR while also discarding some correct matches. A practical threshold should be selected from a separate validation set for an acceptable FPR; selecting and evaluating it on these same pairs would overstate performance. The 5-pixel correctness tolerance and the 3-pixel RANSAC tolerance also affect the ROC.

## AI use disclosure

I wrote the main assignment code. I used OpenAI Codex to help write helper functions and to revise and organize this report.

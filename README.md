# Rice 2023 Spring COMP546/ELEC546 Assignments
-------------------------------------------------------------------------------------------
Rice University 2023 Spring ELEC/COMP 546 or ELEC/COMP 447 (INTRODUCTION TO COMPUTER VISION) Assignments

[Course Website](https://computervision.rice.edu/)

# Instructor

<img src="https://bpb-us-e1.wpmucdn.com/blogs.rice.edu/dist/a/12547/files/2023/01/headshot-273x300.jpg" width="200" height="220" />

**Professor Guha Balakrishnan ([email](guha@rice.edu))**

# Student of this repository
<img src="https://avatars.githubusercontent.com/u/20944449?v=4" width="200" height="200" />

**Shaun Lin ([email](hl116@rice.edu))**

<details><summary>HW1</summary>
<p>

<details><summary>1.0 Basic Image Operations (10 points)</summary>
<p>

In this problem, you will gain some experience working with [NumPy](https://numpy.org/doc/stable/user/index.html#user) and [OpenCV](https://docs.opencv.org/4.x/) to perform basic image manipulations.

### 1.1 Combining Two Images:

a. Read in two large (> 256 x 256) images, A and B into your Colab notebook (see [sample Colab notebook](https://computervision.rice.edu/resources/#:~:text=Basic%20image%20operations) that was shared with the class earlier).

b. Resize A to 256x256 and crop B at the center to 256x256.
    
c. Create a new image C such that the left half of C is the left half of A and the right half of C is the right half of B.
    
d. Using a loop, create a new image D such that every odd numbered row is the corresponding row from A and every even row is the corresponding row from B.

e. Accomplish the same task in part d without using a loop.

### 1.2 Color Spaces

a. Download the peppers image from [this link](https://blogs.mathworks.com/images/loren/173/peppers_BlueHills.png). Return a binary image (only 0s and 1s), with 1s corresponding to only the yellow peppers. Do this by setting a minimum and maximum threshold value on pixel values in the R,G,B channels. Note that you won’t be able to perfectly capture the yellow peppers, but give your best shot!

b. While RGB is the most common color space for images, it is not the only one. For example, one popular color space is HSV (Hue-Saturation-Value). Hue encodes color, value encodes lightness/darkness, and saturation encodes the intensity of the color. For a visual, see Fig. 1 of this [wiki article](https://en.wikipedia.org/wiki/HSL_and_HSV). Convert the image to the HSV color space using OpenCV’s [cvtColor() function](https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/), and try to perform the same task by setting a threshold in the Hue channel.

c. Add both binary images to your report. Which colorspace was easier to work with for this task, and why?

</p>
</details>

<details><summary>2.0 2D Geometric Transforms (15 points)</summary>
<p>

### 2.1 Write functions to produce transformation matrices
Write separate functions that output the 3 x 3 transformation matrices for the following transforms: **translation**, **rotation**, **similarity** (translation, rotation, and scale), and **affine**. The functions should take as input the following arguments:

1. Translation: horizontal and vertical displacements
2. Rotation: angle
3. Similarity: angle, horizontal/vertical displacements, and scale factor (assume equal scaling for horizontal and vertical dimensions)
4. Affine: 6 parameters

The output of each function will be a 3 x 3 matrix.

### 2.2 Write a function that warps an image with a given transformation matrix
Next, write a function imwarp(I, T) that warps image I with transformation matrix T. The function should produce an output image of the same size as I. See Fig. 1 for an example of a warp induced by a rotation transformation matrix. ```Make the origin of the coordinate system correspond to the CENTER of the image, not the top-left corner. This will result in more intuitive results, such as how the image is rotated around its center in Fig. 1. ```

![Fig. 1](https://github.com/PiscesLin/Rice_2023_Spring_ELEC546_Assignments/blob/main/HW1/Input%20image/HW1_description_images_2.2.png)

**Hint 1:** Consider the transformation matrix T to describe the mapping from each pixel in the output image back to the original image. By defining T in this way, you can account for each output pixel in the warp, resulting in no ‘holes’ in the output image (see Lec. 03 slides).

**Hint 2:** What happens when the transformation matrix maps an output pixel to a non-integer location in the input image? You will need to perform bilinear interpolation to handle this correctly (see Lec. 03 slides).

**Hint 3:** You may find NumPy’s [meshgrid function](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) useful to generate all pixel coordinates at once, without a loop.

### 2.3 Demonstrate your warping code on two color images of your choice
For each of the two images, show 2-3 transformations of each type (translation, rotation, similarity, affine) in your report.

</p>
</details>

<details><summary>3.0 Cameras (15 points)</summary>
<p>

### 3.1 Camera Matrix Computation
a. Calculate the camera intrinsic matrix **K**, extrinsic matrix **E**, and full rank 4 ⨉ 4 projection matrix **P = KE** for the following scenario with a pinhole camera:

- The camera is rotated 90 degrees around the x-axis, and is located at (1, 0, 2)^𝑇.
  
- The focal lengths 𝑓𝑥, 𝑓𝑦 are 100.

- The principal point (𝑐𝑥, 𝑐𝑦)^𝑇 is (25, 25).

b. For the above defined projection, find the world point in inhomogeneous coordinates x𝑤 which corresponds to the projected homogeneous point in image space 𝑥𝐼 =
(25, 50, 1, 0.25)^T

</p>
</details>

<details><summary>4.0 Relighting (10 points) (ELEC/COMP 546 ONLY)</summary>
<p>
In this problem, you will perform a simple version of image relighting, the task of changing the
lighting on a scene. To do this experiment, you will need two light sources (such as ceiling
lights, floor lamps, flashlights etc.) and a couple of scene objects. Set up a static scene similar
to the one shown in Fig. 2 (the light sources do not have to be seen in the frame, but try to have
them illuminating the scene at two different angles), and a camera such that it is stationary
throughout the experiment (cell phone leaning against heavy object or wall is fine). Let us label
the two lamps as LAMP1 and LAMP2. 

![Fig. 2](https://github.com/PiscesLin/Rice_2023_Spring_ELEC546_Assignments/blob/main/HW1/Input%20image/HW1_description_images_4.0.png)

a. Capture the image of the scene by turning on LAMP1 only (image I1). Now capture an image by turning on LAMP2 only (image I2). Finally, capture the image with both LAMP1 and LAMP2 on (image I12). Load and display these images into your Colab notebook.

b. Now, you will create a synthetic photo (I12_synth) depicting the scene when both of the lamps are turned on by simply summing I1 and I2 together: I12_synth = I1 + I2. Also compute an image depicting the difference between the synthetic and real images: D = I12_synth - I12.

c. In your report, show I1, I2, I12, I12_synth, and D side by side. When displaying D, make sure to rescale D’s values to fill the full available dynamic range ([0,1] for float, or [0,255] for uint8). You can do this with the following operation: 

**(D - min(D))/(max(D) - min(D)).**

d. How good is your synthetic image compared to the real one? Where do they differ the most?

</p>
</details>

</p>
</details>

<details><summary>HW2</summary>
<p>

<details><summary>1.0 Hybrid Images (10 points)</summary>
<p>
Recall the hybrid image of Albert Einstein and Marilyn Monroe introduced in [1] and reproduced below in Fig. 1. Due to the way your brain processes spatial frequencies, you will see the identity of the image change if you squint or move farther/closer to the image. In this problem, you will create your own hybrid image.

![Fig. 1](https://github.com/PiscesLin/Rice_2023_Spring_ELEC546_Assignments/blob/main/HW2/Input%20image/HW2_description_images_1.0.png)

### Gussian kernel

Implement function **gaussian2D(sigma, kernel_size)** that retutns a 2D gaussian blur kernel, with input argument **sigme** specifying a tuple of x,y scales (standar deviations), and **kernel_size** specifying a tuple of x,y dimensions of the kernel. The kernel should be large enough to include 3 standard deviations per dimension.

### Created Hybrid Images

Choose two images(A and B) of your choice that you will blend with one another and convert them to grayscale (you can use [opencv.cvtColor](https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/)). These images can be of faces, or any other objects. Try to make the objects in the two images occupy roughly the same region in the image (if they don't you can use the **imwarp** function you wrote in Homework 1 to manually align them!)

Construct a hybrid image **C** from **A** (to be seen close-up) and **B** (to be seen far away) as follows: **C = blur(B) + (A-blur(A))**, where **blur** is a function that lowpass filters the image (use the Gaussian kernel you coded in 1.1 for this). Try different values of **sigma** for the Gaussian kernel. How does the amount of blurring affect your perception of the results? In your report, please show your input images labeled clearly as **A** and **B**, and attach the result **C** for a value of **sigma** which you feel demonstrates the illusion the best, at both the original size and a downsampled size. As a sanity check, you should be able to see the identity change when looking at the original and downsampled versions.

### Fourier Spectra

For the **sigma** value you chose in 1.2, show images of the Fourier spectra magnitudes of images **A**, **B**, **blur(B)**, **A-blur(A)**, and **C**. You can get the magnitude of the Fourier spectrum coefficients of an image ‘x’ by running:

```X = numpy.abs(numpy.fftshift(numpy.fft2(x)))```

By default, **numpy.fft2** will place the zero frequency (DC component) of the spectrum at the top left of the image, and so **numpy.fftshift** is used here to place the zero frequency at the center of the image. When displaying the Fourier spectrum with **matplotlib.pyplot.imshow**, the image will likely look black. This is because the DC component typically has a much higher magnitude than all other frequencies, such that after rescaling all values to lie in [0,1], most of the image is close to 0. To overcome this, display the logarithm of the values instead.

</p>
</details>

<details><summary>2.0  Laplacian Blending (15 points)</summary>
<p>

The Laplacian pyramid is a useful tool for many computer vision and image processing applications. One such application is blending sections of different images together, as shown in Fig. 2. In this problem, you will write code that constructs a Laplacian pyramid, and use it to blend two images of your choice together.

![Fig. 2](https://github.com/PiscesLin/Rice_2023_Spring_ELEC546_Assignments/blob/main/HW2/Input%20image/HW2_description_images_2.0.png)

### Gaussian Pyramid

Write a function **gausspyr(I, n_levels, sigma)** that returns a Gaussian pyramid for image **I** with number of levels **n_levels** and Gaussian kernel scale **sigma**. The function should return a list of images, with element **i** corresponding to level **i** of the pyramid. Note that level **0** should correspond to the original image I, and level **n_levels - 1** should correspond to the coarsest (lowest frequency) image.

### Image Blending

Choose two images A and B depicting different objects and resize them to the same shape. You may want to use your **imwarp** function from Homework 1 to align the scales/orientations of the objects appropriately (as was done in the example in Fig. 2) so that the resulting blend will be most convincing. Create a binary mask image **mask** which will have 1s in its left half, and 0s in its right half (called a ‘step’ function). Perform blending with the following operations:

1. Build Laplacian pyramids for **A** and **B**.
2. Build a Gaussian pyramid for **mask**.
3. Build a blended Laplacian pyramid for output image **C** using pyramids of **A**, **B**, and **mask**, where each level 𝑙𝐶 is defined by the equation 𝑙𝐶 = 𝑙𝐴 ∗ 𝑚 + 𝑙𝐵 ∗ (1 − 𝑚).

4. Invert the combined Laplacian pyramid back into an output image **C**.

Show the following in your report: 
(1) Images from all levels of the Laplacian pyramids for **A** and **B**.
(2) Images from all levels of the Gaussian pyramid for **mask**.
(3) Your final blended image **C**.

### (ELEC/COMP 546 Only) Blending two images with a mask other than a step

Laplacian blending is not restricted to only combining halves of two images using a step mask. You can set the mask to any arbitrary function and merge images, as shown in [this example](https://surfertas.github.io/static/img/posts/handeye.png). Demonstrate a Laplacian blend of two new images using a mask other than step.

</p>
</details>

<details><summary>3.0 Pulse Estimation from Video (5 points)</summary>
<p>

You are convinced that your friend Alice is a robot. You don’t have much evidence to prove this because she is quite a convincing human during conversations, except for the fact that she does get very angry if water touches her. One day, you hit upon a plan to figure out this mystery once and for all. You know that a human has a heart which pumps blood, and a robot does not. Furthermore, you read a paper [2] showing that one can estimate heart rate from a video of a human face using very simple computer vision techniques. So the next day, you convince Alice to take this video of herself, [linked here](https://drive.google.com/file/d/1xKNv_HKHl-8ErbglEZY2wLYfVVfvTPSK/view?usp=share_link). You will now need to implement a simple pulse estimation algorithm and run it on the video. Follow these steps:

### 3.1 Read video into notebook and define regions of interest

Upload the video into your Colab environment. Note that it may take several minutes for the upload to complete due to the size of the file. You can then read the video frames into a numpy array using the **read_video_into_numpy** function provided [here](https://colab.research.google.com/drive/1eBfpjdWAtXF3-3R3VM4dleBfuuqZUHvq?usp=sharing).

Using the first video frame, manually define rectangles (row and column boundaries) that capture 1) one of the cheeks and 2) the forehead.


### 3.2 Compute signals

Now compute the average Green value of pixels for all frames for each facial region (cheek, forehead). This gives a 1D signal in time called the Photoplethysmogram (PPG) for each region.

### 3.3 Bandpass filter

It is often useful to filter a signal to a particular band of frequencies of interest (‘pass band’) if we know that other frequencies don’t matter. In this application, we know that a normal resting heart rate for an adult ranges between 60-100 beats per minute (1-1.7 Hz). Apply the **bandpass_filter** function to your signals provided [here](https://colab.research.google.com/drive/1eBfpjdWAtXF3-3R3VM4dleBfuuqZUHvq?usp=sharing). You can set low_cutoff = 0.8, high_cutoff = 3, fs = 30, order = 1. Plot the filtered signals.

### 3.4 Plot Fourier spectra

Plot the Fourier magnitudes of these two signals using the [DFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft), where the x-axis is frequency (in Hertz) and y-axis is amplitude. DFT coefficients are ordered in terms of integer indices, so you will have to convert the indices into Hertz. For each index n = [- N/2, N/2], the corresponding frequency is Fs * n / N, where N is the length of your signal and Fs is the sampling rate of the signal (30 Hz in this case). You can use [numpy.fftfreq](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html) to do this conversion for you.

### 3.5 Estimate Alice’s average pulse rate

A normal resting heart rate for adults ranges between 60-100 beats per minute. What rate does the highest peak in Alice’s Fourier spectrum correspond to? Which facial region provides the cleanest spectrum (the one which has the clearest single peak and low energy elsewhere)? Is Alice likely a human or not?

### 3.6 (ELEC/COMP 546 Only) Find your own pulse

Take a 15-20 second video of yourself using a smartphone, webcam, or personal camera. Your face should be as still as possible, and don’t change facial expressions. Do a similar analysis above as you did with Alice’s video. Show some frames from your video. Was it easier/harder to estimate heart rate compared to the sample video we provided? What was challenging about it?

### References

[1] Oliva, Aude, Antonio Torralba, and Philippe G. Schyns. "Hybrid images." ACM Transactions on Graphics (TOG) 25.3 (2006): 527-532

[2] Poh, Ming-Zher, Daniel J. McDuff, and Rosalind W. Picard. "Non-contact, automated cardiac pulse measurements using video imaging and blind source separation." Optics express 18.10 (2010): 10762-10774.
</p>
</details>

</p>
</details>

<details><summary>HW3</summary>
<p>

<details><summary>1.0 Optical Flow</summary>
<p>

In this problem, you will implement both the Lucas-Kanade and Horn-Schunck algorithms. Your implementations should use a Gaussian pyramid to properly account for large displacements. You can use your pyramid code from Homework 2, or you may simply use Opencv’s [pyrDown](https://theailearner.com/tag/cv2-pyrdown/) function to perform the blur + downsampling. You may also use Opencv’s [Sobel filter](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html) to obtain spatial (x,y) gradients of an image.

### 1.1 Lucas-Kanade (5 points)

Implement the Lucas-Kanade algorithm, and demonstrate tracking points on this [video](https://drive.google.com/file/d/1ZtOw3nMYR_vsNJJin6TNHhq-F1RTYvol/view?usp=share_link).

1. Select corners from the first frame using the [Harris corner detector](SelectcornersfromthefirstframeusingtheHarriscornerdetector.Youcanuse). You can use this command: **corners = cv.cornerHarris(gray_img,2,3,0.04).**

2. Track the points through the entire video by applying Lucas-Kanade between each pair of successive frames. This will yield one ‘trajectory’ per point, with length equal to the number of video frames.
3. Create a gif showing the tracked points overlaid as circles on the original frames. You can draw a circle on an image using cv.circle. You can save a gif with this code:

```
import imageio
imageio.mimsave('tracking.gif', im_list, fps = 10)
```

where im_list is a list of your output images. You can open this gif in your web browser to play it as a video and visualize your results. Show a few frames of the gif in your report, and save the gif in your Google Drive, and place the link to it in your report. Make sure to allow view access to the file!

4. Answerthefollowingquestions:
a. Do you notice any inaccuracies in the point tracking? Where and why?
b. How does the tracking change when you change the local window size used in Lucas-Kanade?

### Horn-Schunck (5 points)

Implement the Horn-Schunck algorithm. Display the flow fields for the ‘Army,’ ‘Backyard,’ and ‘Mequon’ test cases from the Middlebury dataset, [located here](https://vision.middlebury.edu/flow/data/comp/zip/eval-color-twoframes.zip). Consider ‘frame10.png’ as the first frame, and ‘frame11.png’ as the second frame for all cases.

Use this code to display each predicted flow field as a colored image:

```
hsv = np.zeros(im.shape, dtype=np.uint8)
hsv[..., 1] = 255
mag, ang = cv.cartToPolar(flow_x, flow_y)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
out = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
```

### 1.3 ELEC/COMP 546 Only: Improving Horn-Schunck with superpixels (5 points)

Recall superpixels discussed in lecture and described further [in this paper](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/SLIC_Superpixels.pdf). How might you use superpixels to improve the performance of Horn-Schunck? Can you incorporate your idea into the smoothness + brightness constancy objection function? Define any notation you wish to use in the equation. You don’t have to implement your idea in code for this question.

</p>
</details>

<details><summary>2.0 Image Compression with PCA</summary>
<p>

In this problem, you will use PCA to compress images, by encoding small patches in low-dimensional subspaces. Download these two images:

[Test Image 1](https://drive.google.com/file/d/1n2BK1Fn2s0hZ8ZO9127JoRAJWabZwalA/view?usp=sharing)

[Test Image 2](https://drive.google.com/file/d/1XiHesOsu23b26BGMz2cdIV6bhELTESHS/view?usp=sharing)

Do the following steps for each image separately.

### 2.1 Use PCA to model patches (5 points)

Randomly sample at least 1,000 16 x 16 patches from the image. Flatten those patches into vectors (should be of size 16 x 16 x 3). Run PCA on these patches to obtain a set of principal components. Please write your own code to perform PCA. You may use **numpy.linalg.eigh**, or **numpy.linalg.svd** to obtain eigenvectors.

Display the first 36 principal components as 16 x 16 images, arranged in a 6 x 6 grid (Note: remember to sort your eigenvalues and eigenvectors by decreasing eigenvalue magnitude!). Also report the % of variance captured by all principal components (not just the first 36) in a plot, with the x-axis being the component number, and y-axis being the % of variance explained by that component.

### 2.2 Compress the image (5 points)

Show image reconstruction results using 1, 3, 10, 50, and 100 principal components. To do this, divide the image into non-overlapping 16 x 16 patches, and reconstruct each patch independently using the principal components. Answer the following questions:

1. Was one image easier to compress than another? If so, why do you think that is the case?

2. What are some similarities and differences between the principal components for the two images, and your interpretation for the reason behind them?

</p>
</details>

</p>
</details>

## Plagiarism

Plagiarism of any form will not be tolerated. You are expected to credit all sources explicitly. If you have any doubts regarding what is and is not plagiarism, talk to me.

## License
This project is licensed under Rice University

<img src="https://brand.rice.edu/sites/g/files/bxs2591/files/2019-08/190308_Rice_Mechanical_Brand_Standards_Logos-9.png" width="500" height="100" />

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

### 1.1 Combining Two Images:

a. Read in two large (> 256 x 256) images, A and B into your Colab notebook.

b. Resize A to 256x256 and crop B at the center to 256x256.
    
c. Create a new image C such that the left half of C is the left half of A and the right half of C is the right half of B.
    
d. Using a loop, create a new image D such that every odd numbered row is the corresponding row from A and every even row is the corresponding row from B.

e. Accomplish the same task in part d without using a loop.

### 1.2 Color Spaces

a. Return a binary image (only 0s and 1s), with 1s corresponding to only the yellow peppers.

b. Convert the image to the HSV color space using OpenCV’s cvtColor() function, and try to perform the same task by setting a threshold in the Hue channel.

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

**Hint 3:** You may find NumPy’s meshgrid function useful to generate all pixel coordinates at once, without a loop.

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

</p>
</details>

<details><summary>HW3</summary>
<p>

</p>
</details>

## License
This project is licensed under Rice University

<img src="https://brand.rice.edu/sites/g/files/bxs2591/files/2019-08/190308_Rice_Mechanical_Brand_Standards_Logos-9.png" width="500" height="100" />

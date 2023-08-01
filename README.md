# Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch

Using kernel matrixes and other filters to process images in order to detect and track objects.

The following examples will be from scratch in Python using only numpy arrays without relying on computer vision packages such as OpenCv.

A kernel matrix acts as a filter that allows the processing and extraction of features from an image when multiplying the image matrix and the kernel matrix and summing the result:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/0e220ba3-e129-4f5b-a445-1fcdd0fcdac1)

The example shown above utilized a blurring kernel filter in which all the values in the kernel matrix are 1/9 so the output image is the same image but blurred.

There are many different kinds of kernel filters; the kernel shown below can be utilized for detecting the foreground of the image from the background of the image:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/d161104c-3513-444a-930e-0db16ab9b453)


However, the example above is "easy" because the object (i.e. drone) is the only foreground in the image so detecting and tracking that object in Python is simple:

    from skimage import io as io
    image = io.imread('https://i2-prod.cheshire-live.co.uk/incoming/article21305374.ece/ALTERNATES/s615/0_Surfing-Olympics-Day-2.jpg')

    import numpy as np
    def normalize_rgb_values(rgb_values, max_value=1.0):
        norm_rgb_values = (rgb_values - np.mean(rgb_values)) / np.var(rgb_values)**0.5
        norm_rgb_values += abs(np.min(norm_rgb_values))
        norm_rgb_values *= (max_value / np.max(norm_rgb_values))
        return np.round(norm_rgb_values, decimals=0).astype(int) if max_value == 255 else np.round(norm_rgb_values, decimals=9).astype(float)

    image = normalize_rgb_values(image[:, :, :3])

    _kernel = np.diag([-5.0, 2.0, 2.0])
    _kernel[np.where(_kernel == 0.0)] = -1.0

    def pad_image(image, _kernel):
        '''Adds zeros to image border; padded image size = image size + (2 * kernel size)'''
        _image = np.zeros((image.shape[0] + _kernel.shape[0]*2, image.shape[1] + _kernel.shape[1]*2, 3)).astype(image.dtype)
        _image[_kernel.shape[0]:-_kernel.shape[0], _kernel.shape[1]:-_kernel.shape[1], :] = image
        return _image

    image = pad_image(image, _kernel)
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()

    def filter_image(image, _kernel):
        '''Processes image by multiplying it with kernel filter; returns image with higher foreground values and lower background values'''
        _image = np.zeros((image.shape)).astype(float)
        for row in range(image.shape[0] - _kernel.shape[0]):
            for col in range(image.shape[1] - _kernel.shape[1]):
                for rgb in range(image.shape[2]):
                    _image[row, col, rgb] = np.sum(image[row:row + _kernel.shape[0], col:col + _kernel.shape[1], rgb]*_kernel)
        return _image

    filtered_image = normalize_rgb_values(filter_image(image, _kernel))
    plt.imshow(filtered_image)
    plt.show()

First we get the image (obviously), which is a link from an image I found on Google images.

We have to normalize the values if they aren't already, then we pad the image which involves adding zeros to the image's border (increasing its size) and the padding is determined by the size of the kernel.

Then we filter the image by iterating rows, columns, and RGB (third dimension) values of the image matrix multiplying it by the kernel and summing the result.

To actually extract the foreground so that we have a binary image matrix to work with where ones represent the foreground and zeros represent the background, we'll have to process those kernel filtered values:

    def process_filtered_image(filtered_image, summed_threshold=1.5):
        '''Procsses filtered image via a summed binary threshold; return binary image where ones are foreground and zeros are background'''
        processed_filtered_image = np.zeros((filtered_image.shape[0], filtered_image.shape[1])).astype(float)
        processed_filtered_image[np.where(np.sum(filtered_image, axis=2) >= summed_threshold)] = 1.0
        return processed_filtered_image

    processed_filtered_image = process_filtered_image(filtered_image)
    plt.imshow(processed_filtered_image)
    plt.show()


Which gives us:

![image](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/95980485-dd22-466b-8377-97e1718801b1)

Now, the padding of the kernel filtered image caused the image's borders to also have ones; we'll use this to our advantage in the following examples:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/241e8243-6ae6-46c7-8586-27bbddeba163)

After processing the image, we'll include a processing step; this post processing Python function "fattens" up (or thickens) the foreground objects along with the image's borders, thus connecting them:

    def summed_neighbors(processed_filtered_image, row, col, neighbors=10):
        '''Sum of processed filtered image at [row, col] and neighbors (i.e. [row - neighbors:row + neighbors, col - neighbors:col + neighbors])'''
        return np.sum(processed_filtered_image[max(row - neighbors, 0):min(row + neighbors, processed_filtered_image.shape[0]), max(col - neighbors, 0):min(col + neighbors, processed_filtered_image.shape[1])])

    def post_process_filtered_image(processed_filtered_image):
        '''Fattens object silhouette and adds borders that connect with foreground; returns same binary image but now with ones around image border'''
        _processed_filtered_image = processed_filtered_image.copy()
        for row, col in zip(np.where(processed_filtered_image == 0.0)[0], np.where(processed_filtered_image == 0.0)[1]):
            _processed_filtered_image[row, col] = 1.0 if summed_neighbors(processed_filtered_image, row, col) > 1.0 else 0.0
        return _processed_filtered_image

    post_processed_filtered_image = post_process_filtered_image(processed_filtered_image)
    plt.imshow(post_processed_filtered_image)
    plt.show()

The post_process_filtered_image function iterates over the row-column values of the binary image that are zero, and if it has neighboring ones we make it equal to one as well.
So the returned image will be the same as the input binary image except we'll padd the foreground pixels as long as they have have neighboring pixels determined via the summed_neighbors function;
the additional benefit is that "random" pixels or pixels generated from noise we'll be smoothed out--thus, this function both interpolates where pixel values of ones should be, and removes pixel values of ones where they shouldn't be.

And the function did one more thing; it "fattened" up the pixels of ones on the image's border, thus connecting the foreground to the image's border.

This provides us the opportunity to remove the foreground connecting to the image's borders, leaving only the object we want to detect and track:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/64557edd-28b5-4f9d-85aa-3a8a45148f47)


We utilize that the foreground is connected to the image's borders, so where (numpy's where function) these row-column values equal one (representing the foreground including the object) and do not connect to the image's borders (numpy's sum function on the current row-column through the shape of the image) we populate a numpy matrix of zeros with ones:

    def remove_foreground(post_processed_filtered_image):
        '''Removes foreground leaving only the object in the image; returns same binary image but now without the foreground connecting to the borders of image'''
        _object_silhouette_in_image = np.zeros((post_processed_filtered_image.shape)).astype(float)
        for row, col in zip(np.where(post_processed_filtered_image == 1.0)[0], np.where(post_processed_filtered_image == 1.0)[1]):
            if np.sum(post_processed_filtered_image[row:, col]) < post_processed_filtered_image[row:, col].shape[0]*0.5 and np.sum(post_processed_filtered_image[row, col:]) < post_processed_filtered_image[row, col:].shape[0]*0.5:
                _object_silhouette_in_image[row, col] = 1.0
        return _object_silhouette_in_image

    _object_silhouette_in_image = remove_foreground(post_processed_filtered_image)
    plt.imshow(_object_silhouette_in_image)
    plt.show()

This gives us the silhouette of the object without the background and without the foreground.

Applications for computer vision video image processing are wide ranging, and the potential for automating object detection and object tracking that will improve efficiency should be explored.
For example, a Ukrainian tank that needs AA (Anti Air) defenses specifically against small russian military drones would require software for processing video camera feed and first and foremost be able to discriminate between background and foreground, friendly nearby soldiers and tanks from incoming enemy drones, etc.(I know the people playing soccer in the image aren't soldiers but pretend they are):

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/b0f51f29-1bf2-433b-95ec-06a2cc9d2dbb)


Taking a break, to be continued.













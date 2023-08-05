# Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch

Using kernel matrixes and other video image processing filters to detect and track objects; simply put, the computer vision techniques we'll use will be for removing the background from images and then removing the foreground apart from the object--specifically images where the object is NOT (or at least not entirely) in the foreground but regardless of the color of the object and without having to input or adjust any context-dependent parameters.

The following examples will be from scratch in Python using only numpy arrays without relying on computer vision packages such as OpenCv, so we're doing all the math from scratch. We'll mainly use numpy's where function and numpy's sum function, so nothing fancy--we're just relying on simple math for figuring out the background from the foreground, and the foreground from the object in video images which means this computer vision Python model generalizes across video images (e.g. the object could be lighter colors, darker colors, a variety of colors or just one color, small or large in size, background and foreground similar to the object or not homogenous, etc.

Here's the goal of this repository visualized in the figure below, including the steps numbered in order:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/802aaf56-89e2-48e8-a471-556da7a706d1)


A kernel matrix acts as a filter that allows the processing and extraction of features from an image when multiplying the image matrix and the kernel matrix and summing the result:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/0e220ba3-e129-4f5b-a445-1fcdd0fcdac1)

The example shown above utilized a blurring kernel filter in which all the values in the kernel matrix are 1/9 so the output image is the same image but blurred.

There are many different kinds of kernel filters; the kernel shown below can be utilized for detecting the foreground of the image from the background of the image by increasing foreground (including the object we want to detect and track) RGB values and decreasing background RGB values:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/d161104c-3513-444a-930e-0db16ab9b453)


However, the example above is "easy" because the object (i.e. the drone) is the only foreground in the image so detecting and tracking that object in Python isn't that difficult:

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
        _padded_image = np.zeros((image.shape[0] + _kernel.shape[0]*2, image.shape[1] + _kernel.shape[1]*2, 3)).astype(image.dtype)
        _padded_image[_kernel.shape[0]:-_kernel.shape[0], _kernel.shape[1]:-_kernel.shape[1], :] = image
        return _padded_image

    image = pad_image(image, _kernel)
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()

    def filter_image(image, _kernel):
        '''Filters image by multiplying it with kernel matrix; returns image with higher foreground values and lower background values'''
        _filtered_image = np.zeros((image.shape)).astype(float)
        def get_rgb_sums(image, _kernel):
            '''returns the sums for each of the RGB channels'''
            return np.array([np.sum(image[:, :, rgb]*_kernel) for rgb in range(image.shape[2])]).astype(float)

        for row in range(image.shape[0] - _kernel.shape[0]):
            for col in range(image.shape[1] - _kernel.shape[1]):
                _filtered_image[row, col, :] = get_rgb_sums(image[row:row + _kernel.shape[0], col:col + _kernel.shape[1], :], _kernel)
        return _filtered_image

    filtered_image = normalize_rgb_values(filter_image(image, _kernel))
    plt.imshow(filtered_image)
    plt.show()


First we get the image (obviously), which is a link from an image I found on Google images.

We have to normalize the values if they aren't already, then we pad the image which involves adding zeros to the image's border--increasing its size by two times the size of the kernel.

Then we filter the image by iterating rows and columns and getting the RGB (third dimension) values of the image matrix multiplying it by the kernel and summing the result.

To actually extract the foreground so that we have a binary 1-channel image matrix to work with where ones represent the foreground and zeros represent the background, we'll have to process those kernel filtered values:

    def process_filtered_image(filtered_image, summed_threshold=1.5):
        '''Procsses filtered image via a summed binary threshold; returns binary 1-channel image where ones are foreground and zeros are background'''
        processed_filtered_image = np.zeros((filtered_image.shape[0], filtered_image.shape[1])).astype(float)
        processed_filtered_image[np.where(np.sum(filtered_image, axis=2) >= summed_threshold)] = 1.0
        if processed_filtered_image[0, 0] == 0.0:
            processed_filtered_image -= 1.0
            processed_filtered_image[np.where(processed_filtered_image == -1.0)] = 1.0
        return processed_filtered_image

    processed_filtered_image = process_filtered_image(filtered_image)
    plt.imshow(processed_filtered_image)
    plt.show()


Which gives us:

![image](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/95980485-dd22-466b-8377-97e1718801b1)

Now, the padding of the kernel filtered image caused the image's borders to also have ones--this is an advantageous side effect; we'll go over how to use this to our advantage in the following examples:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/c8d9a526-4341-456e-b163-5a6bc172eca2)


Our goal is to discriminate the object from the foreground in the image (now that we have the 1-channel binary image with ones representing the foreground and the background is zeros), so in the post processing step we enhance or exacerbate these features to the extent that the foreground that isn't the object connects to the borders of the image:

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/ff31840a-f1a9-4399-8572-78151772306b)


This post processing Python function "fattens" up (or thickens) the foreground objects along with the image's borders, thus connecting them (but not connecting to the object we want to detect and track):

    def post_process_filtered_image(processed_filtered_image, neighbors=20):
        '''Fattens object silhouette and adds borders that connect with foreground; returns same binary image but now with ones around image border'''
        _processed_filtered_image = processed_filtered_image.copy()

        def summed_neighbors(processed_filtered_image, row, col, neighbors):
            '''Sum of processed filtered image at [row, col] and neighbors (i.e. [row - neighbors:row + neighbors, col - neighbors:col + neighbors])'''
            return np.sum(processed_filtered_image[max(row - neighbors, 0):min(row + neighbors, processed_filtered_image.shape[0]), max(col - neighbors, 0):min(col + neighbors, processed_filtered_image.shape[1])])

        for row, col in zip(np.where(processed_filtered_image == 0.0)[0], np.where(processed_filtered_image == 0.0)[1]):
            _processed_filtered_image[row, col] = 1.0 if summed_neighbors(processed_filtered_image, row, col, neighbors) > 1.0 else 0.0
        return _processed_filtered_image

    post_processed_filtered_image = post_process_filtered_image(processed_filtered_image)
    plt.imshow(post_processed_filtered_image)
    plt.show()


The post_process_filtered_image function iterates over the row-column values of the binary 1-channel image that are equal to zero, and if it has neighboring ones we make it equal to one as well.
So the returned image will be the same as the input binary image except we'll exacerbate the foreground pixels as long as they have have neighboring pixels determined via the inner summed_neighbors function;
the additional benefit is that "random" pixels or pixels generated from noise we'll be smoothed out--thus, this function both interpolates where pixel values of ones should be, but also avoids padding ones where they shouldn't be.

Since the foreground can now be discriminated by whether it connects to the borders of the image or not, we have the opportunity to remove the foreground connecting to the image's borders--leaving only the object we want to detect and track.

We utilize that the foreground is connected to the image's borders, so where (i.e. numpy's where function) these row-column values equal one (representing the foreground including the object) and do not connect to the image's borders (i.e. numpy's mean function on the current row-column through the shape of the image) we populate a numpy matrix of zeros with ones (four times... in the third dimension... and the minimum sum of those four silhouettes THAT ISN'T ZERO is our final image... after we smooth it):

    def remove_foreground(post_processed_filtered_image, image, summed_threshold=0.5):
        '''Removes foreground leaving only the object in the image (input image already has background removed); returns same binary image but now without the foreground connecting to the borders of image
        the _object_silhouette_in_image matrix includes both forward and backward passes through the foreground of the image and those values are stored in the third dimension--the minimum sum of values that isn't zero is the object silhouette,
        which we then smooth (i.e. we draw a box around it);
        row, col, post_processed_filtered_image: forward passes;
        _row, _col, _post_processed_filtered_image: backward passes;
        Two if-statements discriminate whether foreground contacts the borders of image or not (NOT being the object);
        '''
        _object_silhouette_in_image = np.zeros((post_processed_filtered_image.shape[0], post_processed_filtered_image.shape[1], 4)).astype(float)
        _post_processed_filtered_image = np.flip(np.flip(post_processed_filtered_image, axis=0), axis=1)
        for row, col, _row, _col in zip(np.where(post_processed_filtered_image == 1.0)[0], np.where(post_processed_filtered_image == 1.0)[1], np.where(_post_processed_filtered_image == 1.0)[0], np.where(_post_processed_filtered_image == 1.0)[1]):
            if np.mean(post_processed_filtered_image[row:, col]) < summed_threshold and np.mean(post_processed_filtered_image[row, col:]) != 1.0 or np.mean(post_processed_filtered_image[row, col:]) < summed_threshold and np.mean(post_processed_filtered_image[row:, col]) != 1.0:
                _object_silhouette_in_image[row, col, 0] = 1.0
            if np.mean(_post_processed_filtered_image[_row:, _col]) < summed_threshold and np.mean(_post_processed_filtered_image[_row, _col:]) != 1.0 or np.mean(_post_processed_filtered_image[_row, _col:]) < summed_threshold and np.mean(_post_processed_filtered_image[_row:, _col]) != 1.0:
                _object_silhouette_in_image[_row, _col, 1] = 1.0
            if np.mean(post_processed_filtered_image[row:, col]) < summed_threshold and np.mean(post_processed_filtered_image[row, col:]) < summed_threshold:
                _object_silhouette_in_image[row, col, 2] = 1.0
            if np.mean(_post_processed_filtered_image[_row:, _col]) < summed_threshold and np.mean(_post_processed_filtered_image[_row, _col:]) != 1.0 or np.mean(_post_processed_filtered_image[_row, _col:]) < summed_threshold and np.mean(_post_processed_filtered_image[_row:, _col]) != 1.0:
                _object_silhouette_in_image[_row, _col, 3] = 1.0

        def get_min_summed_object(_object_silhouette_in_image):
            '''returns the minimum summed object silhouette that isn't zero'''
            _min_sum_nums = [_object for _object in range(_object_silhouette_in_image.shape[2]) if np.sum(_object_silhouette_in_image[:, :, _object]) != 0.0]
            _object_sums_list = [_object_silhouette_in_image[:, :, 0], np.flip(np.flip(_object_silhouette_in_image[:, :, 1], axis=0), axis=1), _object_silhouette_in_image[:, :, 2], np.flip(np.flip(_object_silhouette_in_image[:, :, 3], axis=0), axis=1)]
            _min_sum = [np.sum(_object_silhouette_in_image[:, :, _object]) for _object in _min_sum_nums]
            return _object_sums_list[np.where(_min_sum == np.min(_min_sum))[0][0]]

        def smooth_object_silhouette(_object_silhouette_in_image):
            '''returns image with a smoothed silhouette of the object; basically, it draws a box around the silhouette of the object'''
            _smoothed_object_silhouette_in_image = _object_silhouette_in_image.copy()
            rows, cols = np.where(_object_silhouette_in_image == 1.0)
            _smoothed_object_silhouette_in_image[np.min(rows):np.max(rows), np.min(cols):np.max(cols)] = 1.0
            return _smoothed_object_silhouette_in_image

        return smooth_object_silhouette(get_min_summed_object(_object_silhouette_in_image))

    _object_silhouette_in_image = remove_foreground(post_processed_filtered_image, image)
    plt.imshow(_object_silhouette_in_image)
    plt.show()


This gives us the silhouette of the object without the background and without the foreground in the image.

How the remove_foreground function works is by making both a forward and a backward pass through the post processed filtered image where (i.e. numpy's where function) it equals one (i.e. the foreground), and if the row-column arrays do not connect to the borders, we populate that pixel with a value of one; we're carrying out two forward passes and two backward passess each iteration of its loop--and the values are in the third dimension of the matrix which has a size of four (values for the two forward and two backwards passes).

The "forward" pass means we use the original, unmanipulated image while the "backward" pass means the twice flipped image (i.e. numpy's flip function over the 0th axis and then again over the 1st axis). The "pass" refers to the row-column values as we iterate through the image (shown as red arrows in the figure below):

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/0f199f03-7811-4255-abc3-b6598f274590)


The minimum sum of the four silhouettes (that doesn't equal zero... if/when it does sum to zero then that pass wasn't appropriate so we drop it/don't consider it) is our final silhouette of the image with only the object present and represented as ones in the matrix which is found with the inner get_min_summed_object function. The last component of the remove_foreground function calls the inner smooth_object_silhouette function to re-pad the object and it essentially draws a box around the object we wanted to be detected and tracked.

The final thing (which isn't really an image processing step since we already have our binary 1-channel image containing only the object's silhouette with the background and foreground removed from the image) is to draw a bounding box around the object in original image:

    def outline_object_in_image(image, _object_silhouette_in_image, outline_color=None, outline_width=1):
        '''returns the original image but now with a bounding box around the object'''
        if outline_color is None:
            outline_color = np.array([1.0, 1.0, 1.0]).astype(float) - image[0, 0, :]

        rows, cols = np.where(_object_silhouette_in_image == 1.0)
        fill_rows = np.arange(np.min(rows), np.max(rows), 1).astype(int)
        fill_cols = np.arange(np.min(cols), np.max(cols), 1).astype(int)

        _object_outlined_in_image = image.copy()
        _object_outlined_in_image[fill_rows, np.min(cols) - outline_width:np.min(cols) + outline_width, :] = outline_color
        _object_outlined_in_image[fill_rows, np.max(cols) - outline_width:np.max(cols) + outline_width, :] = outline_color
        _object_outlined_in_image[fill_rows, int(np.round((np.max(cols) + np.min(cols))*0.5, decimals=0)) - outline_width:int(np.round((np.max(cols) + np.min(cols))*0.5, decimals=0)) + outline_width, :] = outline_color

        _object_outlined_in_image[np.min(rows) - outline_width:np.min(rows) + outline_width, fill_cols, :] = outline_color
        _object_outlined_in_image[np.max(rows) - outline_width:np.max(rows) + outline_width, fill_cols, :] = outline_color
        _object_outlined_in_image[int(np.round((np.max(rows) + np.min(rows))*0.5, decimals=0)) - outline_width:int(np.round((np.max(rows) + np.min(rows))*0.5, decimals=0)) + outline_width, fill_cols, :] = outline_color
        return _object_outlined_in_image

    _object_outlined_in_image = outline_object_in_image(image, _object_silhouette_in_image, outline_color=np.array([1.0, 0.0, 0.0]).astype(float))
    plt.imshow(_object_outlined_in_image)
    plt.show()


Applications for computer vision video image processing are wide ranging, and the potential for automating object detection and object tracking that will improve human-operated task efficiency should be explored.

![Picture1](https://github.com/OriYarden/Computer-Vision-Image-Processing-Object-Detection-Tracking-in-Python-from-Scratch/assets/137197657/b0f51f29-1bf2-433b-95ec-06a2cc9d2dbb)


Limitations: although this computer vision video image processing guide can work on a range of different images, but for detecting and tracking objects the object must be at least not entirely in the foreground of the image.

Practical Applications: this computer vision video image processing guide can work on a range of different images regardless of the object's colors, sizes, etc. and its resiliient in that it works without requiring the user to input values or adjust parameters; it also doesn't require training.


"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    #Okay so let me think about this. We have the MNIST file format.
    #And it looks something like this.
    """
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
    """
    #So the first 16 Bytes are the header.
    #And then we have the data.
    #Okay. What's with the number of rows and the number of columns?
    #Oh yeah, that's how many pixels are there in every image.
    #So each image is Row*Column long.
    #Okay so slowly but surely, a picture is forming.
    with gzip.open(image_filesname) as image_bin:
        img_header = image_bin.read(16)
        img_magic_number, number_of_images, rows, cols = struct.unpack('>IIII',img_header)
        print("number of colums is ", cols)
        print("number of rows is ", rows)
        print("number of images is ", number_of_images)
        print("magic number is ", img_magic_number)
        #Okay so now I have an open file and I can do something about it. 
        #The header has been parsed. I have the data that I need.
        #Or atleast the metadata that I need. Now I Need to pack each image in a numpy array.
        #How do I do that?
        #I don't think that I'm supposed to use a for-loop. Right? How do you populate an array anyway?
        #So maybe something like preinitializing the array and then populating the fields?
        #What I have found is even more efficient.
        images = np.frombuffer(image_bin.read(number_of_images*rows*cols),dtype=np.uint8)
        images = images.reshape(number_of_images, rows*cols).astype(np.float32)/255


    with gzip.open(label_filename) as label_bin:
        lbl_header = label_bin.read(8)
        lbl_magic_number, number_of_items = struct.unpack('>II',lbl_header)
        print("magic_number for labels is ", lbl_magic_number)
        print("number of items is ", number_of_items)

        labels = np.frombuffer(label_bin.read(number_of_items),dtype=np.uint8)

    (X,y) = (images,labels)
    print(y[0:10])
    return (X,y)


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])

    So we already have the logit predictions for each class.
    y is a vector of one-hot encodings of the true label of each class.
    
    Lce = -log(softmax(Zi,yi))
    
    """
    #I really think that python's list comprehension might be the way to go
    #So lets say that we're computing individual losses L_i
    #Compute individual losses L_i for each example
    #individual_losses = [-np.log(np.exp(Z[i, y[i]]) / np.sum(np.exp(Z[i]))) for i in range(batch_size)]
    #Compute the average loss over the batch
    #This is literally the formula from the HW1 ipynb
    #Yes, I am confused slightly but essentially, we have 
    #Loss =  (log(summation(exp(zi))) - zy )/ batch_size
    return (ndl.log(ndl.exp(Z).sum((1,))).sum() - (y_one_hot * Z).sum()) / Z.shape[0]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    for start_idx in range(0, X.shape[0], batch):
        # Create batches
        # X_batch = ndl.Tensor(X[start_idx:start_idx + batch])
        # y_batch = y[start_idx:start_idx + batch]

        # # Compute logits 
        # Z1 = ndl.matmul(X_batch,W1) #this is the hypothesis h from the lectures.
        # Z2 = ndl.matmul(ndl.relu(Z1),W2)
        # # Create a one-hot encoding of the true labels
        # Iy = np.zeros_like(Z2.realize_cached_data())
        # #print("Iy shape is ",Iy.shape)
        # Iy[np.arange(X_batch.shape[0]), y_batch] = 1
        # print("Iy is", Iy.shape)
        # Iy = ndl.Tensor(Iy)
        # print("Iy tensor is", Iy.shape)
        # # Softmax
        # LCE= softmax_loss(Z2,Iy)
        # print("LCE is", LCE.shape)
        # # Compute the gradient
        # batch_gradient = LCE.backward()
        # print(batch_gradient)
        # # Update theta

        iterations = (y.size + batch - 1) // batch
        for i in range(iterations):
            x = ndl.Tensor(X[i * batch : (i+1) * batch, :])
            Z = ndl.relu(x.matmul(W1)).matmul(W2)
            yy = y[i * batch : (i+1) * batch]
            y_one_hot = np.zeros((batch, y.max() + 1))
            y_one_hot[np.arange(batch), yy] = 1
            y_one_hot = ndl.Tensor(y_one_hot)
            loss = softmax_loss(Z, y_one_hot)
            loss.backward()
            W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
            W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
        return W1, W2

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

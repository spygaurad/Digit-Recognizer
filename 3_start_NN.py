# nueral network skeleton
import numpy
from scipy.special import expit
import matplotlib.pyplot as ppl
from scipy import misc


class NeuralNetwork:

    # initialization of NN
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # initialize weights
        # numpy.random.normal randomizes values with mean 0 and standard deviance 1/root of number of links to the node
        # this is defined in book as a better method  that numphy.random.rand

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # sigmoid function
        self.activation_function = lambda x: expit(x)
        pass

    # train the NN
    def train(self, input_list, target_list):

        # convert inputs to 2D array  .T denotes transpose
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # links into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # links out of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # links or signals involving output layer
        outputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(outputs)

        # backpropagation error
        output_error = targets - final_outputs
        hidden_error = numpy.dot(self.who.T, output_error)

        #updating weight
        self.who += self.lr * numpy.dot((output_error * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_error * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))
        pass

    # query the NN
    def query(self, input_list):

        # convert inputs to 2D array  .T denotes transpose
        inputs = numpy.array(input_list, ndmin=2).T

        # links into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # links out of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # links or signals involving output layer
        outputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(outputs)

        return final_outputs


innodes = 784
hnodes = 300
onnodes = 10
learning_rate = 0.1
n = NeuralNetwork(innodes, hnodes, onnodes, learning_rate)

# data = open("Dataset/mnist_train_100.csv", 'r')
data = open("Dataset/mnist_train.csv", 'r')
data_list = data.readlines()
data.close()
epoch = 1
for i in range(epoch):
    for each_value in data_list:

        each_image = each_value.split(',')
        array_image = numpy.asfarray(each_image[1:])

        # scale input to range 0 to 1
        # scaled_image = (numpy.asfarray(each_image[1:])/255.0 * 0.9)+0.1
        # print(scaled_image)
        scaled_image = array_image * 0.99 / 255 + 0.1
        # ppl.imshow(scaled_image.reshape(28, 28), cmap='Greys', interpolation=None)
        # ppl.show()

        targets = numpy.zeros(onnodes) + 0.1
        targets[int(each_image[0])] = 0.99
        n.train(scaled_image, targets)
        pass
    pass

#
# # test cases list
# accuracy = []
# # query_data = open("Dataset/mnist_test_10.csv", 'r')
# query_data = open("Dataset/mnist_test.csv", 'r')
# test_list = query_data.readlines()
# query_data.close()
# for values in test_list:
#     value = values.split(',')
#     out_value = int(value[0])
#     input_query = numpy.asfarray(value[1:]) * 0.99 / 255 + 0.1
#     got_matrix = n.query(input_query)
#     max_in_matrix = numpy.argmax(got_matrix)
#
#     if(out_value == max_in_matrix):
#         accuracy.append(1)
#     else:
#         accuracy.append(0)
#         pass
#     pass
#
# accuracy_array = numpy.asarray(accuracy)
# print("Performance: ", accuracy_array.sum() / accuracy_array.size)


# # convert image to greyscale png
# img = Image.open('Dataset/5.jpg').convert('LA')
# img = img.resize((28,28), Image.ANTIALIAS)
# img.save('Dataset/5new.png')

img_array = misc.imread('Dataset/2.png', flatten=True)   # flatten image to points, turn color to greyscale

# because mnist datasset reads 0 as white which is usually black so we reverse
img_data = 255.0 - img_array.reshape(784)
img_d = (img_data / 255.0 * 0.99) + 0.01
predict_value = n.query(img_d)
print(predict_value)
val = numpy.argmax(predict_value)
print(val)

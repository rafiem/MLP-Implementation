#!/usr/bin/python3
import json
import csv
import math
import matplotlib.pyplot as plt
import random

class MLP():
  def __init__(self, alpha, filename, test_size, seed):
    self.parsing_data(filename, test_size, seed)
    self.alpha          = alpha
    self.akurasi_train  = []
    self.error_train    = []
    self.akurasi_test   = []
    self.error_test     = []

  
  def set_thetas(self, theta_input, theta_hidden):
    self.theta_input  = theta_input
    self.theta_hidden = theta_hidden

  
  def set_bias(self, bias_input, bias_hidden):
    self.bias_input   = bias_input
    self.bias_hidden  = bias_hidden


  def parsing_data(self, file_name, test_size, seed):
    f = open(file_name)
    csv_obj = csv.DictReader(f, fieldnames = ( "x1","x2","x3","x4","name"))
    parsed_data = eval(json.dumps([row for row in csv_obj]))
    
    for item in parsed_data:
      item["data"] = [float(item["x{}".format(i)]) for i in range(1,5)]
      for i in range(1,5): 
        item.pop("x{}".format(i))
      if item["name"] == "Iris-setosa":
        item["type"] = [0,0]
      elif item["name"] == "Iris-versicolor":
        item["type"] = [1,0]
      elif item["name"] == "Iris-virginica":
        item["type"] = [0,1]
    
    random.seed(seed)
    random.shuffle(parsed_data)
    self.train_data = parsed_data[:test_size]
    self.test_data = parsed_data[test_size:]
    f.close()
    return parsed_data


  def result(self, flower_data, modified_theta, modified_bias):
    return sum(list(x*y for x,y in zip(flower_data,modified_theta))) + modified_bias


  def sigmoid(self, modified_result):
    return 1/(1+math.exp(-(modified_result)))


  def get_error(self, flower_type, modified_activation):
    return pow(int(flower_type) - modified_activation, 2)


  def check_prediction(self, target, output):
    for i in range(len(output)):
      if(output[i] > 0.5):
          data = 1
      else:
          data = 0
      if(data != target[i]):
          return False
    return True


  def calculate_delta_error(self, data, out_o, count):
    delta_err = 0.0
    for i in range(len(out_o)):
      delta_err += (out_o[i] - data['type'][i]) * out_o[i] * (1 - out_o[i]) * self.theta_hidden[i][count]
    
    return delta_err


  def update_theta_and_bias_input(self, data, out_h, delta_err, count):
    for i in range(len(self.theta_input)):
      delta = delta_err * out_h[count] * (1-out_h[count]) * data["data"][i]
      self.theta_input[count][i] -= self.alpha * float(delta)
    
    delta = delta_err * out_h[count] * (1-out_h[count])
    self.bias_input[count] -= self.alpha * float(delta)


  def update_theta_and_bias_hidden(self, data, out_o, out_h, count):
    for i in range(len(self.theta_hidden[count])):
      delta = (out_o[count] - data["type"][count]) * out_o[count] * (1-out_o[count]) * out_h[i]
      self.theta_hidden[count][i] -= self.alpha * float(delta)

    delta = (out_o[count] - data["type"][count]) * out_o[count]*(1-out_o[count])
    self.bias_hidden[count] -= self.alpha * float(delta)
    

  def training(self):
    akurasi = 0
    error   = 0

    for item_data in self.train_data:
      output_hidden = [ self.sigmoid(self.result(item_data["data"], self.theta_input[i], self.bias_input[i])) for i in range(len(self.theta_input))]
      output_out    = [ self.sigmoid(self.result(output_hidden, self.theta_hidden[i], self.bias_hidden[i])) for i in range(len(self.theta_hidden))]

      for i in range(len(self.theta_input)):
        delta_error = self.calculate_delta_error(item_data, output_out, i)
        self.update_theta_and_bias_input(item_data, output_hidden, delta_error, i)

      for i in range(len(self.theta_hidden)):
        self.update_theta_and_bias_hidden(item_data, output_out, output_hidden, i)

      error += sum(map(lambda x,y: self.get_error(x, y), item_data['type'], output_out))
      if(self.check_prediction(item_data['type'], output_out)):
        akurasi += 1

    self.error_train.append(error / len(self.train_data))
    self.akurasi_train.append(akurasi / len(self.train_data))

  def testing(self):
    akurasi = 0
    error   = 0

    for item_data in self.test_data:
      output_hidden = [ self.sigmoid(self.result(item_data["data"], self.theta_input[i], self.bias_input[i])) for i in range(len(self.theta_input))]
      output_out    = [ self.sigmoid(self.result(output_hidden, self.theta_hidden[i], self.bias_hidden[i])) for i in range(len(self.theta_hidden))]

      error += sum(map(lambda x,y: self.get_error(x, y), item_data['type'], output_out))
      if(self.check_prediction(item_data['type'], output_out)):
        akurasi += 1

    self.error_test.append(error / len(self.test_data))
    self.akurasi_test.append(akurasi / len(self.test_data))  

  def plotting(self):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(self.akurasi_train, color='red', label='Train')
    plt.plot(self.akurasi_test, color='blue', label='Test')
    plt.title('Accuracy, learning rate : {}'.format(self.alpha))
    plt.xlabel("Epoch")
    plt.ylabel("Avg Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(self.error_train, color='red', label='Train')
    plt.plot(self.error_test, color='blue', label='Test')
    plt.title('Loss, learning rate : {}'.format(self.alpha))
    plt.xlabel("Epoch")
    plt.ylabel("Avg Error")
    plt.legend()
    plt.grid(True)

    plt.show()
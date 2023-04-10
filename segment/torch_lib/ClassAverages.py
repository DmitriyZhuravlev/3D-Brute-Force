import numpy as np
import os
import json


"""
Enables writing json with numpy arrays to file
"""
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self,obj)

"""
Class will hold the average dimension for a class, regressed value is the residual
"""
class ClassAverages:
    def __init__(self, classes=[]):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/class_averages.txt'

        if len(classes) == 0: # eval mode
            self.load_items_from_file()

        for detection_class in classes:
            class_ = detection_class.lower()
            if class_ in self.dimension_map.keys():
                continue
            self.dimension_map[class_] = {}
            self.dimension_map[class_]['count'] = 0
            self.dimension_map[class_]['total'] = np.zeros(3, dtype=np.double)


    def add_item(self, class_, dimension):
        class_ = class_.lower()
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['total'] += dimension
        # self.dimension_map[class_]['total'] /= self.dimension_map[class_]['count']

    def get_item(self, class_):
        class_ = class_.lower()
        #return self.dimension_map[class_]['total'] / self.dimension_map[class_]['count']
        
        #label : Cyclist
        # dim : [1.76       0.54166667 1.78666667]
        
        
        
        # -------------
        # label : Car
        # dim : [1.511875   1.62678571 3.79705357]
        
        
        # label : Truck
        # dim : [ 3.09        2.38714286 10.70571429]
        
        
        # -------------
        # label : Car
        # dim : [1.511875   1.62678571 3.79705357]
        
        
        # label : Pedestrian
        # dim : [1.77611111 0.65944444 0.83666667]
    
        if class_ == "pedestrian":
            return [1.77611111, 0.65944444, 0.83666667]
        if class_ == "car":
            return [1.511875,   1.62678571, 3.79705357]
        if class_ == "truck":
            #return [ 3.09,        2.38714286, 10.70571429]
            return [3.09, 2.6 , 16.1544]
        if class_ == "cyclist":
            return [1.76,       0.54166667, 1.78666667]
        return [0,0,0]

    def dump_to_file(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        f.close()

    def load_items_from_file(self):
        f = open(self.filename, 'r')
        dimension_map = json.load(f)

        for class_ in dimension_map:
            dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

        self.dimension_map = dimension_map

    def recognized_class(self, class_):
        return class_.lower() in self.dimension_map

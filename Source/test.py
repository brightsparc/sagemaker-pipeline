import json
import sys
import time
from sagemaker.predictor import RealTimePredictor

start = time.time()

configuration_file = sys.argv[1]

with open(configuration_file) as f:
    data = json.load(f)

endpoint_name = data["Parameters"]["EndpointName"]

print('Starting testing endpoint: {}'.format(endpoint_name))

pred = RealTimePredictor(endpoint_name)
pred.content_type = 'text/csv'
pred.accept = 'text/csv'

# Expecting class 4
test_values = "80,1056736,3,4,20,964,20,0,6.666666667,11.54700538,964,0,241.0,482.0,931.1691850999999,6.6241710320000005,176122.6667,\
431204.4454,1056315,2,394,197.0,275.77164469999997,392,2,1056733,352244.3333,609743.1115,1056315,24,0,0,0,0,72,92,\
2.8389304419999997,3.78524059,0,964,123.0,339.8873763,115523.4286,0,0,1,1,0,0,0,1,1.0,140.5714286,6.666666667,\
241.0,0.0,0.0,0.0,0.0,0.0,0.0,3,20,4,964,8192,211,1,20,0.0,0.0,0,0,0.0,0.0,0,0,20,2,2018,1,0,1,0"

result = int(pred.predict(test_values))
print(result)
assert result==4

# Expecting class 7
test_values = "80,10151,2,0,0,0,0,0,0.0,0.0,0,0,0.0,0.0,0.0,197.0249237,10151.0,0.0,10151,10151,10151,10151.0,0.0,10151,10151,0,0.0,\
0.0,0,0,0,0,0,0,40,0,197.0249237,0.0,0,0,0.0,0.0,0.0,0,0,0,0,1,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2,0,0,0,32738,\
-1,0,20,0.0,0.0,0,0,0.0,0.0,0,0,21,2,2018,2,0,1,0"

result = int(pred.predict(test_values))
print(result)
assert result==7

# Expecting class 0
test_values = "80,54322832,2,0,0,0,0,0,0.0,0.0,0,0,0.0,0.0,0.0,0.0368169318,54322832.0,0.0,54322832,54322832,54322832,54322832.0,0.0,\
54322832,54322832,0,0.0,0.0,0,0,0,0,0,0,40,0,0.0368169318,0.0,0,0,0.0,0.0,0.0,0,0,0,0,1,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,\
0.0,0.0,0.0,0.0,2,0,0,0,279,-1,0,20,0.0,0.0,0,0,0.0,0.0,0,0,23,2,2018,4,0,1,0"

result = int(pred.predict(test_values))
print(result)
assert result==0

end = time.time()
print('Test complete in {}'.format(end - start))

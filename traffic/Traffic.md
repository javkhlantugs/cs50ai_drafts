# Traffic

## Abstract

 - As a starting point, I basically copied the CNN structure from the lecture with a single convolution and a single Max Pooling layer and tested various layer structures from there.

## Steps

1. - 1 Convolutional layer (32 Filters, 3x3 kernel with 3 channel value)
   - 1 MaxPooling layer (2x2)
   - 1 Dropout layer to avoid overfitting (Rate 0.5)
   - Result: 0.0821

2. - 1 Convolutional layer (32 Filters, 3x3 kernel with 3 channel value)
   - 1 MaxPooling layer (2x2)
   - 1 more Convolutional layer (32 Filters, 3x3 kernel with 3 channel value)
   - 1 Dropout layer to avoid overfitting (Rate 0.5)
   - Result: 0.9425

3. - 1 Convolutional layer (32 Filters, 3x3 kernel with 3 channel value)
   - 1 MaxPooling layer (2x2)
   - 1 more Convolutional layer (32 Filters, 3x3 kernel with 3 channel value)
   - 1 more Maxpooling layer (2x2)
   - 1 Dropout layer to avoid overfitting (Rate 0.5)
   - Result: 0.9566

4. - 1 Dropout layer rate changed to (Rate 0.6)
   - Result: 0.9609

5. - 1 Dropout layer rate changed again to (Rate 0.7)
   - Result: 0.0548

6. - Bringing the dropout rate to 0.4
   - Result: 0.0548

## Conclusion
The first step with just one convolutional layer and one Maxpooling layer, the result was terrible with accuracy less than ten percent. Adding another convolutional layer made a huge improvement bumping the accuracy to more than 90% but the difference in accuracy were insignificant after other additions and tweeking. Increasing the dropout rate to more than 0.6 significantly decreased the performance.


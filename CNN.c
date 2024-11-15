/*
    Author: Jacob Ludwigson

    Notes for possible future optimization as needed: Reverse Filters are stored in a way in which there is an instance of each filter AND its reverse, for every single filter.
        -This could be altered to be some sort of stack local variable, however, I have designed it this way such that they are dynamically created based on architechture parameters.
        -A solution that is potentially better would be to malloc memory for one filter at each "layer" the iterator passes in the outer loop of the convolutional weight updates.
*/
#include <math.h>
#include <stdio.h>  // file handling functions
#include <stdlib.h> // atoi
#include <string.h> // strtok
#include <float.h>
#include <time.h>
typedef struct image {
    double* grayscale;
    short label;
}image;
// Yes this is a lot of globals, no I do not care. You try to do this task passing around a gazillion double pointers to every function as parameters.
int numConvolutionalLayers;
int* numFiltersAtConvolutionalLayers;
int* filterSizes;
int* filterStrides;

double*** filters;
double*** reverseFilters;
double** filterBiases;

double*** convolutionalLayers;
double** paddedImages;
double** paddedGradients;
double** summedImgs;
double** maxPooledLayers;
int** pooledIndices;
double** pooledGradients;
double*** convolutionalGradients;
int* sizeConvolutionalLayers;

double* flattenedImgs; 
double* flattenedLoss;

int numFullyConnectedLayers;
int* numNodesAtFullyConnectedLayers;

double** fullyConnectedZ;
double** fullyConnectedActivated;
double*** fullyConnectedWeights;
double** fullyConnectedBiases;
double** fullyConnectedGradients;

double leakyAlpha;
//We may want to randomize the weights/biases matrices/vectors here.
void randInitArray(double* array, int size, double lower, double upper) {
    for (int i = 0; i < size; i++) {
        array[i] = lower + ((double)rand() / RAND_MAX) * (upper - lower); // Scale to [1, 20]
    }
}
double rand_normal() {
    static int hasSpare = 0;
    static double spare;

    if (hasSpare) {
        hasSpare = 0;
        return spare;
    }

    hasSpare = 1;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}

// Function to initialize weights using He Normal initialization
void he_normal_init(double *weights, int size, int fan_in) {
    double scale = sqrt(2.0 / fan_in);
    for (int i = 0; i < size; i++) {
        weights[i] = rand_normal() * scale;
    }
}
void initCNN(){
    int necPadding;
    int paddedImgWidth;
    int imgWidth;
    srand(time(NULL));
    filters = malloc(numConvolutionalLayers * sizeof(double*));
    reverseFilters = malloc(numConvolutionalLayers * sizeof(double*));
    convolutionalLayers = malloc(numConvolutionalLayers * sizeof(double*));
    convolutionalGradients = malloc(numConvolutionalLayers * sizeof(double*));
    maxPooledLayers = malloc(numConvolutionalLayers * sizeof(double*));
    pooledIndices = malloc(numConvolutionalLayers * sizeof(int*));
    pooledGradients = malloc(numConvolutionalLayers * sizeof(double*));
    filterBiases = malloc(numConvolutionalLayers * sizeof(double*));
    paddedImages = malloc(numConvolutionalLayers * sizeof(double *));
    paddedGradients = malloc(numConvolutionalLayers * sizeof(double));
    summedImgs = malloc(numConvolutionalLayers * sizeof(double *));

    for (int i = 0; i < numConvolutionalLayers; i++){
        filters[i] = malloc(numFiltersAtConvolutionalLayers[i] * sizeof(double*));
        reverseFilters[i] = malloc(numFiltersAtConvolutionalLayers[i] * sizeof(double*));
        convolutionalLayers[i] = malloc(numFiltersAtConvolutionalLayers[i] * sizeof(double*));
        convolutionalGradients[i] = malloc(numFiltersAtConvolutionalLayers[i] * sizeof(double*));
        imgWidth = sqrt(sizeConvolutionalLayers[i]);
        necPadding = filterSizes[i] / 2;
        paddedImgWidth = imgWidth + 2 * necPadding;
        paddedImages[i] = malloc(sizeof(double) * paddedImgWidth * paddedImgWidth);
        paddedGradients[i] = malloc(sizeof(double) * paddedImgWidth * paddedImgWidth);
        for (int j = 0; j < numFiltersAtConvolutionalLayers[i]; j++){ 
            filters[i][j] = malloc(filterSizes[i] * filterSizes[i]  * sizeof(double));
            reverseFilters[i][j] = malloc(filterSizes[i] * filterSizes[i]  * sizeof(double));
            he_normal_init(filters[i][j], filterSizes[i]*filterSizes[i], filterSizes[i]*filterSizes[i]);
            // randInitArray(filters[i][j], filterSizes[i]*filterSizes[i], -0.15f, 0.15f);
            convolutionalLayers[i][j] = malloc(sizeConvolutionalLayers[i] * sizeof(double));
            convolutionalGradients[i][j] = malloc(sizeConvolutionalLayers[i] * sizeof(double));
            
        }
        pooledGradients[i] = malloc(sizeConvolutionalLayers[i]/4 * sizeof(double));
        summedImgs[i] = malloc(sizeof(double) * sizeConvolutionalLayers[i]);
        maxPooledLayers[i] = malloc(sizeConvolutionalLayers[i]/4 * sizeof(double));
        pooledIndices[i] = malloc(sizeConvolutionalLayers[i]/4 * sizeof(int));
        filterBiases[i] = malloc(sizeof(double) * numFiltersAtConvolutionalLayers[i]);
        randInitArray(filterBiases[i], numFiltersAtConvolutionalLayers[i], 0.0f, 0.1f);
    }

    flattenedImgs = (double* ) malloc(sizeof(double) * sizeConvolutionalLayers[numConvolutionalLayers-1]/4);
    flattenedLoss = (double *) malloc(sizeof(double) * sizeConvolutionalLayers[numConvolutionalLayers-1]/4);

    fullyConnectedActivated = malloc(sizeof(double*) * numFullyConnectedLayers);
    fullyConnectedWeights = malloc(sizeof(double*) * numFullyConnectedLayers);
    fullyConnectedGradients = malloc(sizeof(double*) * numFullyConnectedLayers);
    fullyConnectedZ = malloc(sizeof(double*) * numFullyConnectedLayers);
    fullyConnectedBiases = malloc(sizeof(double*) * numFullyConnectedLayers);


    int sizePrevLayer = sizeConvolutionalLayers[numConvolutionalLayers-1]/4;
    for (int i = 0; i < numFullyConnectedLayers; i++){
        fullyConnectedActivated[i] = (double*) malloc(sizeof(double) * numNodesAtFullyConnectedLayers[i]);
        if (i != 0){
            sizePrevLayer = numNodesAtFullyConnectedLayers[i-1];
        }
        fullyConnectedWeights[i] = malloc(sizeof(double*) * numNodesAtFullyConnectedLayers[i]);
        for (int j = 0; j < numNodesAtFullyConnectedLayers[i]; j++){
            fullyConnectedWeights[i][j] = malloc(sizeof(double) * sizePrevLayer);
            he_normal_init(fullyConnectedWeights[i][j],sizePrevLayer,numNodesAtFullyConnectedLayers[i]);
            // randInitArray(fullyConnectedWeights[i][j], sizePrevLayer, -0.15, 0.15);
        }
        fullyConnectedGradients[i] = (double*) malloc(sizeof(double) * numNodesAtFullyConnectedLayers[i]);
        memset(fullyConnectedGradients[i], 0.0, sizeof(double) * numNodesAtFullyConnectedLayers[i]);
        fullyConnectedZ[i] = (double*) malloc(sizeof(double) * numNodesAtFullyConnectedLayers[i]);
        fullyConnectedBiases[i] = (double*) malloc(sizeof(double) * numNodesAtFullyConnectedLayers[i]);
        randInitArray(fullyConnectedBiases[i], numNodesAtFullyConnectedLayers[i], 0.0, 0.1);
    }
}
void destructCNN(){
    for (int i = 0; i < numConvolutionalLayers; i++){
        for (int j = 0; j < numFiltersAtConvolutionalLayers[i]; j++){
            free(reverseFilters[i][j]);
            free(filters[i][j]);
            free(convolutionalLayers[i][j]);
            free(convolutionalGradients[i][j]);
        }
        free(filters[i]);
        free(reverseFilters[i]);
        free(paddedImages[i]);
        free(paddedGradients[i]);
        free(convolutionalLayers[i]);
        free(convolutionalGradients[i]);
        free(maxPooledLayers[i]);
        free(pooledIndices[i]);
        free(pooledGradients[i]);
        free(filterBiases[i]);
        free(summedImgs[i]);
    }
    free(summedImgs);
    free(paddedImages);
    free(paddedGradients);
    free(filters);
    free(reverseFilters);
    free(convolutionalLayers);
    free(convolutionalGradients);
    free(maxPooledLayers);
    free(pooledIndices);
    free(pooledGradients);
    free(filterBiases);

    free(flattenedImgs);
    free(flattenedLoss);

    for (int i = 0; i < numFullyConnectedLayers; i++){
        free(fullyConnectedActivated[i]);
        for (int j = 0; j < numNodesAtFullyConnectedLayers[i]; j++){
            free(fullyConnectedWeights[i][j]);
        }
        free(fullyConnectedWeights[i]);
        free(fullyConnectedGradients[i]);
        free(fullyConnectedZ[i]);
        free(fullyConnectedBiases[i]);
    }
    free(fullyConnectedActivated);
    free(fullyConnectedWeights);
    free(fullyConnectedGradients);
    free(fullyConnectedZ);
    free(fullyConnectedBiases);
}
void printMatrix(double* matrix, int sizeX, int sizeY) {
    // Loop through each row of the matrix
    for (int i = 0; i < sizeX; i++) {
        // Loop through each column in the current row
        for (int j = 0; j < sizeY; j++) {
            // Print each element with a fixed width for better alignment
            printf("%.5f ", matrix[i * sizeY + j]);  // Adjust the precision as needed
        }
        printf("\n");  // Newline after each row
    }
}
void printImg(double* grayscale, int upperBound, int imageWidth, int label){
    int j = 0;
    int val;
    for (int i = 0; i < upperBound; i++){
        j = i % imageWidth;
        val = grayscale[i] * 255.0;
        printf("%d",val);
        if (val < 100  && val >= 0){
            printf(" ");
        }
        if (val < 10  && val >= 0){
            printf(" ");
        }
        if (j== (imageWidth - 1)){
            printf("\n");
        }
    }
    printf("LABEL : %d\n", label);
}
double getValofPix(double* image, double* filter, int imageWidth, int pixel, int filterSize) {
    double result = 0.0;
    int halfFilter = filterSize / 2;

    // Loop over the filter dimensions
    for (int i = -halfFilter; i <= halfFilter; ++i) {
        for (int j = -halfFilter; j <= halfFilter; ++j) {
            // Calculate the index in the filter and image based on offsets
            int filterIndex = (i + halfFilter) * filterSize + (j + halfFilter);
            int imageRow = pixel / imageWidth + i;
            int imageCol = pixel % imageWidth + j;
            int imageIndex = imageRow * imageWidth + imageCol;

            // Boundary check to avoid out-of-bounds access
            if (imageRow >= 0 && imageRow < imageWidth && imageCol >= 0 && imageCol < imageWidth) {
                // printf("Accessing: imageIndex: %d\n", imageIndex);
                // printf("Accessing: filterIndex: %d\n", filterIndex);
                result += image[imageIndex] * filter[filterIndex];
            }
        }
    }

    return result;
}
void getValofFilter(double* image, double* gradients, double* filter, int imageWidth, int pixel, int filterSize, double learningRate, double* filterBias, int filterNumber) {
    int halfFilter = filterSize / 2;
    // double biasGradientSum = 0.0; // Accumulator for bias gradient
    double gradientSum = 0.0;
    double value;
    // Loop over the filter dimensions
    for (int i = -halfFilter; i <= halfFilter; ++i) {
        for (int j = -halfFilter; j <= halfFilter; ++j) {
            // Calculate the index in the filter and image based on offsets
            int filterIndex = (i + halfFilter) * filterSize + (j + halfFilter);
            int imageRow = pixel / imageWidth + i;
            int imageCol = pixel % imageWidth + j;
            int imageIndex = imageRow * imageWidth + imageCol;

            // Boundary check to avoid out-of-bounds access
            if (imageRow >= 0 && imageRow < imageWidth && imageCol >= 0 && imageCol < imageWidth) {
                // printf("Accessing: imageIndex: %d\n", imageIndex);
                // printf("Accessing: filterIndex: %d\n", filterIndex);
                value = image[imageIndex] > 0.0 ? 1 : leakyAlpha;
                filter[filterIndex] = filter[filterIndex] - learningRate * (gradients[imageIndex] * value);
                gradientSum += gradients[imageIndex];
            }
        }
    }
    filterBias[filterNumber] = filterBias[filterNumber] - (learningRate * gradientSum);
}
void filterUpdate(double * image, double* gradients, double* filter, int stride, int imgWidth, int filterSize, double learningRate, double* filterBias, int filterNumber){
    int necPadding = filterSize / 2;

    int paddedImgWidth = imgWidth + 2 * necPadding;
    int paddedImgSize = paddedImgWidth * paddedImgWidth;
    int iter = 0;
    int outputIter = 0;
    int regPixelIndex;
    for (int i = necPadding; i < paddedImgWidth - necPadding ; i += stride) {
        for (int j = necPadding; j < paddedImgWidth - necPadding; j += stride) {
            int pixelIndex = i * paddedImgWidth + j;
            getValofFilter(image, gradients, filter, imgWidth, pixelIndex, filterSize, learningRate, filterBias, filterNumber);

        }
    }
    //Need to think about how to pass filterBias
    // filterBias[filterNumber] = filterBias[filterNumber] - (learningRate * gradientSum);
}
void convolute(double* image, double* retImg, double* paddedImgs, double* filter, int stride, int imgWidth, int filterSize, double filterBias){
    int necPadding = filterSize / 2;

    int paddedImgWidth = imgWidth + 2 * necPadding;
    int paddedImgSize = paddedImgWidth * paddedImgWidth;

    int iter = 0;
    // printImg(paddedImgs, paddedImgSize, paddedImgWidth, 1);
    int outputIter = 0;
    int regPixelIndex;
    int pixelValue;
    memcpy(retImg,image, imgWidth*imgWidth*sizeof(double));
    for (int i = necPadding; i < paddedImgWidth - necPadding ; i += stride) {
        for (int j = necPadding; j < paddedImgWidth - necPadding; j += stride) {
            int pixelIndex = i * paddedImgWidth + j;
            regPixelIndex = (i - necPadding) * imgWidth + (j - necPadding);
            pixelValue = getValofPix(paddedImgs, filter, paddedImgWidth, pixelIndex, filterSize) + filterBias;
            //apply relu activation
            retImg[regPixelIndex] = pixelValue > 0.0 ? pixelValue : leakyAlpha * pixelValue;
        }
    }
    // printf("PRINTING RETURN IMAGE\n");
    // printImg(retImg, imgWidth*imgWidth, imgWidth, 1);
}
void addTwoMatrices(double* A, double* B, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            A[i * size + j] += B[i*size + j];
        }
    }
}
void maxPool(double** image, int* indices, double* summedImg,  double* returnImage, int upperBound, int imgWidth, int howManyFiltersAtPrevConvLayer){
    double* A;
    //there is no fucking way this is right.
    for (int f = 0; f < howManyFiltersAtPrevConvLayer; f++){
        A = image[f];
        addTwoMatrices(summedImg, A, imgWidth);
    }
    int stride = 2;
    int iter = 0;
    for (int i = 0; i < upperBound - imgWidth; i += (imgWidth*2)) {  // Move by two rows (skip)
        for (int j = i; j < imgWidth + i; j += stride) {  // Move by two columns (skip)
            // Perform the pooling on a 2x2 block
            returnImage[iter] = fmax(fmax(fmax(summedImg[j],summedImg[j + 1]),summedImg[j + imgWidth]),summedImg[j + imgWidth + 1]);
            if (returnImage[iter] == summedImg[j]){
                indices[iter] = 0;
            }
            else if (returnImage[iter] == summedImg[j + 1]){
                indices[iter] = 1;
            }
            else if (returnImage[iter] == summedImg[j+imgWidth]){
                indices[iter] = 2;
            }  
            else if (returnImage[iter] == summedImg[j + imgWidth + 1]){
                indices[iter] = 3;
            }
            iter += 1;
        }
    }
    return;
}
void deMaxPool(double** imagesAtLayer, double** imageGradients,double* maxPooledGradients,int* maxPooledIndices,double* summedImage, int imgWidth, int numFiltersAtLayer){
    int stride = 2;
    int iter = 0;
    int lowerBound = 0;
    int upperBound = imgWidth*imgWidth;
    double mult;
    for (int k = 0; k < numFiltersAtLayer; k++){
        iter = 0;
        for (int i = lowerBound; i < upperBound - imgWidth; i += (imgWidth*2)) {  // Move by two rows (skip)
            for (int j = i; j < imgWidth + i; j += stride) {  // Move by two columns (skip)
                // Perform the pooling on a 2x2 block
                if (summedImage[j] == 0){
                    continue;
                }
                if (maxPooledIndices[iter] == 0){
                    imageGradients[k][j] = (maxPooledGradients[iter]/summedImage[j]) * imagesAtLayer[k][j];
                }
                else if (maxPooledIndices[iter] == 1){
                    imageGradients[k][j + 1] = (maxPooledGradients[iter]/summedImage[j]) * imagesAtLayer[k][j];
                }
                else if (maxPooledIndices[iter] == 2){

                    imageGradients[k][j + imgWidth] = (maxPooledGradients[iter]/summedImage[j]) * imagesAtLayer[k][j];
                }
                else if (maxPooledIndices[iter] == 3){
                    imageGradients[k][j + imgWidth + 1] = (maxPooledGradients[iter]/summedImage[j]) * imagesAtLayer[k][j];
                }
                iter += 1;
            }
        }
    }

    return;
}

int classifySample(image* Img, int startingSize){
    int filtersAtPrevLayer = 0;
    int flatSizeAcc = 0;
    int imgWidth;
    int necPadding;
    int paddedImgWidth;
    int paddedImgSize;
    int iter = 0;
    memcpy(convolutionalLayers[0][0],Img->grayscale,sizeConvolutionalLayers[0] * sizeof(double));
    for (int i = 0; i < numConvolutionalLayers; i++){
        imgWidth = sqrt(sizeConvolutionalLayers[i]);
        necPadding = filterSizes[i] / 2;
        paddedImgWidth = imgWidth + 2 * necPadding;
            // double* summedImg = calloc(upperBound,sizeof(double));
        iter = 0;
        memset(paddedImages[i], 0.0, paddedImgWidth * paddedImgWidth * sizeof(double));
        if( i == 0) {
            memcpy(convolutionalLayers[i][0],Img->grayscale,sizeConvolutionalLayers[i] * sizeof(double));
            continue;
        } 
        else if (i == 1){
            for (int k = 0; k < imgWidth; k++) {
                for (int g = 0; g < imgWidth; g++) {
                    int paddedIndex = (k + necPadding) * paddedImgWidth + (g + necPadding);
                    paddedImages[i][paddedIndex] = convolutionalLayers[i-1][0][iter];
                    iter++;
                }
            }
        }
        else{
            for (int k = 0; k < imgWidth; k++) {
                for (int g = 0; g < imgWidth; g++) {
                    int paddedIndex = (k + necPadding) * paddedImgWidth + (g + necPadding);
                    paddedImages[i][paddedIndex] = maxPooledLayers[i-1][iter];
                    iter++;
                }
            }
        }
        for (int j = 0; j < numFiltersAtConvolutionalLayers[i]; j++){
            if (i == 1){
                if (!convolutionalLayers[i-1][0] || !convolutionalLayers[i][j] || !filters[i][j] || !filterStrides[i] || !numFiltersAtConvolutionalLayers[i-1] || !filterSizes[i] || !filterBiases[i][j]){
                    printf("Null pointers in convolutional layers for some reason!\n");
                    destructCNN();
                    exit(0); 
                    return 0;
                }
                convolute(convolutionalLayers[i-1][0],convolutionalLayers[i][j],paddedImages[i], filters[i][j], filterStrides[i], imgWidth, filterSizes[i], filterBiases[i][j]);
            }
            else{
                convolute(maxPooledLayers[i-1],convolutionalLayers[i][j],paddedImages[i], filters[i][j], filterStrides[i], imgWidth, filterSizes[i], filterBiases[i][j]);
            }
            // printf("Printing convoluted image\n");
            // printImg(convolutionalLayers[i][j], imgWidth*imgWidth, imgWidth,0);
        }
        if (i != 0){
            memset(summedImgs[i], 0.0, sizeConvolutionalLayers[i] * sizeof(double));
            maxPool(convolutionalLayers[i], pooledIndices[i],summedImgs[i], maxPooledLayers[i], sizeConvolutionalLayers[i], imgWidth,numFiltersAtConvolutionalLayers[i-1]);
            // printf("Printing Max Pooled Image, Size: %d\n",sizeConvolutionalLayers[i]/4);
            // printImg(maxPooledLayers[i],sizeConvolutionalLayers[i]/4, imgWidth/2, 0);
            if (i == numConvolutionalLayers-1){
                memcpy(flattenedImgs, maxPooledLayers[i], sizeof(double) * sizeConvolutionalLayers[i]/4);
                flatSizeAcc+=sizeConvolutionalLayers[i]/4;
            }
        }
    }
    // printf("Flattened Images\n");
    // printMatrix(flattenedImgs, 1, sizeConvolutionalLayers[numConvolutionalLayers-1]/4);
    int sizePrevLayer = flatSizeAcc;
    double* prevLayerPtr = flattenedImgs;
    double currSum = 0.0;
    for (int i = 0; i < numFullyConnectedLayers; i++){
        if (i != 0){
            sizePrevLayer = numNodesAtFullyConnectedLayers[i-1];
            prevLayerPtr = fullyConnectedActivated[i-1];
        }
        for (int j = 0; j < numNodesAtFullyConnectedLayers[i]; j++){

            fullyConnectedZ[i][j] = 0.0;
            fullyConnectedActivated[i][j] = 0.0;
            currSum = 0.0;
            
            for (int k =0; k < sizePrevLayer; k++){

                currSum += fullyConnectedWeights[i][j][k] * (prevLayerPtr[k]);
                if (fullyConnectedWeights[i][j][k] != fullyConnectedWeights[i][j][k] || isinf(fullyConnectedWeights[i][j][k])){
                    printf("Invalid value at %d currSum: %f\n",j, fullyConnectedWeights[i][j][k]);
                    destructCNN();
                    exit(0);
                }       
                if (isinf(prevLayerPtr[k]) || !(prevLayerPtr[k] == prevLayerPtr[k])){
                    printf("Invalid value at prevLayerPtr layer: %f\n", prevLayerPtr[k]);
                    destructCNN();
                    exit(0);
                }
                // printf("[Curr Sum: %f]\n", currSum);
            }
            // printf("[CurrSum: %f]\n", currSum);
            fullyConnectedZ[i][j] = currSum + fullyConnectedBiases[i][j];
 
            if (i == numFullyConnectedLayers-1){
                
            }
            else{ 
                if (fullyConnectedZ[i][j] != fullyConnectedZ[i][j] || isinf(fullyConnectedZ[i][j])){
                    printf("Invalid value at %d Z layer: %f\n",j, fullyConnectedZ[i][j]);
                    destructCNN();
                    exit(0);
                }
                fullyConnectedActivated[i][j] = fmax(0.0, fullyConnectedZ[i][j]);
                if (fullyConnectedActivated[i][j] <= 0){
                    fullyConnectedActivated[i][j] = leakyAlpha * fullyConnectedZ[i][j];
                }
                if (isinf(fullyConnectedActivated[i][j]) || !(fullyConnectedActivated[i][j] == fullyConnectedActivated[i][j])){
                    printf("Invalid value at %d activated layer: %f\n",j+1, fullyConnectedActivated[i][j]);
                    destructCNN();
                    exit(0);
                }
            }
        }
    }
    double max = -DBL_MAX;
    double expSum = 0.0;
    // double temperature = 100.0f;
    for (int i = 0; i < numNodesAtFullyConnectedLayers[numFullyConnectedLayers-1]; i++) {
        max = fmax(max, fullyConnectedZ[numFullyConnectedLayers-1][i]);
    }
    for (int i = 0; i < numNodesAtFullyConnectedLayers[numFullyConnectedLayers-1]; i++) {
        double expVal = exp(fullyConnectedZ[numFullyConnectedLayers-1][i]);
        // printf("Exp expVal: %f\n", expVal);
        // printf("Exp Sum: %f\n", expSum);
        if (isnan(expSum)){
            exit(0);
        }
        expSum += expVal;
    }
    if (isinf(max)) {
        printf("Cannot normalize, max is infinite\n");
        exit(0);
    }
    double maxProb = -INFINITY;
    double logSumExp = max + log(expSum);
    int index;
    for (int i = 0; i< numNodesAtFullyConnectedLayers[numFullyConnectedLayers-1]; i++){
        fullyConnectedActivated[numFullyConnectedLayers-1][i] = (exp(fullyConnectedZ[numFullyConnectedLayers-1][i])/expSum);
        if ((expSum != expSum || isinf(expSum))){
            printf("ExpSum non defined value: %f\n", expSum);
            destructCNN();
            exit(0);
        }
        else if (fullyConnectedActivated[numFullyConnectedLayers-1][i] != fullyConnectedActivated[numFullyConnectedLayers-1][i] || isinf(fullyConnectedActivated[numFullyConnectedLayers-1][i])){
            printf("Output non defined value: %f\n", fullyConnectedActivated[numFullyConnectedLayers-1][i]);
            destructCNN();
            exit(0);
        }
        // printf("[%d, %f]\n", i,fullyConnectedActivated[numFullyConnectedLayers-1][i]);
        maxProb = fmax(fullyConnectedActivated[numFullyConnectedLayers-1][i], maxProb);
        if (maxProb == fullyConnectedActivated[numFullyConnectedLayers-1][i]){
            index = i;
        }
        // printf("[%d : %f]\n", i, fullyConnectedActivated[numFullyConnectedLayers-1][i]);
    }
    // sleep(1);
    return index;
}
/*
    reverseFilter(double* filter, double* revFilter, int filterSize)
    Def: Takes two same-size double arrays, one to be reversed, and one to store the reversed array in.
    Parameters: filter - filter to be reversed, revFilter - filter to store the reversed filter in, filterSize - Size of filter or sqrt(filterLength)
*/
void reverseFilter(double* filter, double* revFilter, int filterSize){
    int upper = (filterSize*filterSize)-1;
    int iIter = 0;
    for (int i = upper; i > -1; i--){
        revFilter[iIter] = filter[i];
        iIter+=1;
    }
}
/*
    Notes for possible future optimization as needed: Reverse Filters are stored in a way in which there is an instance of each filter AND its reverse, for every single filter.
        -This could be altered to be some sort of stack local variable, however, I have designed it this way such that they are dynamically created based on architechture parameters.
        -A solution that is potentially better would be to malloc memory for one filter at each "layer" the iterator passes in the outer loop of the convolutional weight updates.
*/
void backPropogate(image* Img, double learningRate){
    //reset gradients to zero
    for (int i = 0; i < numFullyConnectedLayers; i++){
        memset(fullyConnectedGradients[i], 0.0, sizeof(double) * numNodesAtFullyConnectedLayers[i]);
    }
    memset(flattenedLoss, 0.0, sizeof(double) * sizeConvolutionalLayers[numConvolutionalLayers-1]/4);
    for (int i= 0; i < numConvolutionalLayers; i++){
        for (int j = 0; j < numFiltersAtConvolutionalLayers[i]; j++){
            memset(convolutionalGradients[i][j], 0.0, sizeConvolutionalLayers[i] * sizeof(double));
        }
        memset(pooledGradients[i],0.0, sizeConvolutionalLayers[i]/4 * sizeof(double));
    }




    double labelledOutput[10];
    //Since our output of our network is a vector, create a vector for the actual label
    for (int i = 0; i < numNodesAtFullyConnectedLayers[numFullyConnectedLayers-1]; i++){
        labelledOutput[i] = 0.0f;
        if (i == Img->label){
            labelledOutput[i] = 1.0f;
        }
        //So the derivative of the softmax function essentially boils down to negative residuals, so heres our output layers loss.
        fullyConnectedGradients[numFullyConnectedLayers-1][i] = fullyConnectedActivated[numFullyConnectedLayers-1][i] - labelledOutput[i];
    }
    // printMatrix(fullyConnectedGradients[numFullyConnectedLayers-1], 1, numNodesAtFullyConnectedLayers[numConvolutionalLayers-1]);


    int sizeForwardLayer = numNodesAtFullyConnectedLayers[numFullyConnectedLayers-1];
    double value;
    for (int i = numFullyConnectedLayers-1; i > -1; i--){
        if (i !=numFullyConnectedLayers-1 ){
        for (int j = 0; j < numNodesAtFullyConnectedLayers[i+1]; j++){
            for (int k =0; k < numNodesAtFullyConnectedLayers[i]; k++){
                value = (fullyConnectedActivated[i][k] > 0.0) ? 1.0 : leakyAlpha;
                fullyConnectedGradients[i][k] += (fullyConnectedGradients[i+1][j] * fullyConnectedWeights[i+1][j][k]) * value;
                fullyConnectedWeights[i+1][j][k] = fullyConnectedWeights[i+1][j][k] - (learningRate * (fullyConnectedGradients[i+1][j] * fullyConnectedActivated[i][k]));
            }
            fullyConnectedBiases[i+1][j] = fullyConnectedBiases[i+1][j] - learningRate * fullyConnectedGradients[i+1][j];
        }
        }
        // // // sleep(1);
        // printf("Layer %d Gradient:\n", i);
        // printMatrix(fullyConnectedGradients[i], 1, numNodesAtFullyConnectedLayers[i]);
    }
        

    //We now propogate the gradients through the flattened layer.
    for (int j = 0; j < numNodesAtFullyConnectedLayers[0]; j++){
        for (int k = 0; k < sizeConvolutionalLayers[numConvolutionalLayers-1]/4; k++){
            flattenedLoss[k] += (fullyConnectedGradients[0][j] * fullyConnectedWeights[0][j][k]);
            fullyConnectedWeights[0][j][k] = fullyConnectedWeights[0][j][k] - (learningRate * (fullyConnectedGradients[0][j] * flattenedImgs[k]));
        }
        fullyConnectedBiases[0][j] = fullyConnectedBiases[0][j] - learningRate * fullyConnectedGradients[0][j];
    }

    // printMatrix(flattenedLoss, 1, sizeConvolutionalLayers[numConvolutionalLayers-1]/4);
    //Its time to get freaky, we gotta propogate through the convolutional layers now. (Im going insane |-O:O-|)
    memcpy(pooledGradients[numConvolutionalLayers-1], flattenedLoss, sizeConvolutionalLayers[numConvolutionalLayers-1]/4);
    int imgWidth;
    int necPadding;
    int paddedImgWidth;
    int iter;
    for (int i = numConvolutionalLayers-1; i > 0; i--){

        //defining how a convolutional gradient is generated is a function of 
        //1. deMaxPooling the input from the previous layer
            //A deMaxPool has to take into account that we used a summation of the input across the filter dimension, so we have to somehow undo this 
            // while giving each filtered img its "portion" of the gradient. Sounds kind of hard, but I'd win.
        //For a given filter j, we scale it by dividing all elements by the summed image and then multiplying by the filtered image j.
        imgWidth = sqrt(sizeConvolutionalLayers[i]);
        necPadding = filterSizes[i] / 2;
        paddedImgWidth = imgWidth + 2 * necPadding;

        deMaxPool(convolutionalLayers[i], convolutionalGradients[i], pooledGradients[i], pooledIndices[i], summedImgs[i], imgWidth, numFiltersAtConvolutionalLayers[i]);
        for (int j = 0; j < numFiltersAtConvolutionalLayers[i]; j++){
            //2. Applying a reverse convolution on the forward layer's gradients matrix to populate the gradient matrix of this layer
                //To do this we need to first get ourselves a reverse filter:
                reverseFilter(filters[i][j],reverseFilters[i][j], filterSizes[i]);
                //Now we apply it to our gradient matrix

                //Erm this is awkward, I think the padded image is a wee bit funky...We may have to create a padded image of the filter specific gradients at this layer.
                memset(paddedImages[i], 0.0, paddedImgWidth * paddedImgWidth * sizeof(double));
                memset(paddedGradients[i], 0.0, paddedImgWidth * paddedImgWidth * sizeof(double));
                iter = 0;
                for (int k = 0; k < imgWidth; k++) {
                    for (int g = 0; g < imgWidth; g++) {
                        int paddedIndex = (k + necPadding) * paddedImgWidth + (g + necPadding);
                        paddedGradients[i][paddedIndex] = convolutionalGradients[i][j][iter];
                        paddedImages[i][paddedIndex] = convolutionalLayers[i][j][iter];
                        iter++;
                    }
                }
                //If we are not on the last layer (i.e. the one before the actual image) backpropogate the gradient.
                if (i != 1){
                    convolute(convolutionalGradients[i][j],convolutionalGradients[i][j],paddedGradients[i], filters[i][j], filterStrides[i], imgWidth, filterSizes[i], 0.0);
                    memcpy(pooledGradients[i-1], convolutionalGradients[i][j], sizeConvolutionalLayers[i]);
                }
                //3. Updating filter weights by applying the same filter operation, but this time updating the weights at each index instead of pixels in the return img
                //THERE IS A PROBLEM HERE, I NEED TO CREATE A PADDED GRADIENT AND A PADDED IMAGE FOR THIS LOGIC TO WORK!
                //IN ORDER TO CALCULATE FILTER VALUES I NEED TO HAVE THE ACTIVATED FUNCTION AT THIS NODE WHICH IS THE IMAGE, BUT ALSO THE GRADIENT AT THE PIXEL
                //TO COMBINE THESE AND UPDATE FILTER WEIGHTS I NEED THEM TO BE THE SAME SIZE AND PADDED!
                filterUpdate(paddedImages[i], paddedGradients[i],filters[i][j], filterStrides[i], imgWidth, filterSizes[i], learningRate, filterBiases[i], j);

            //4. Dunking on the opposition
        }
    }

}
void trainCNN(char* filename, int learn, double learningRate){
    image *Img = (image *) malloc(sizeof(image));
    FILE *trainingcsv = NULL;
    char* buffer = malloc(6000);
    int lineNumber = 0;
    double correct = 0.0f;
    double total = 0.0f;
    int prediction;
    short iter = -1;
    short firstpass = 0;
    int num = 9999999;
    trainingcsv = fopen(filename, "r");
    Img->grayscale = malloc(1024 * sizeof(double));
    if (trainingcsv == NULL){
        printf("read Failure!\n");
        return;
    }
    else {
        while (fgets(buffer, 6000, trainingcsv)) {
            iter = -1;
            if (firstpass == 0){
                firstpass = 1;
                continue;
            }
        lineNumber+=1;
        char *field = strtok(buffer, ",");
        for (int i = 0; i < 784; i++){
            Img->grayscale[i] = 0;
        }
        while (field) {
            num = atoi(field); 
            if (iter == -1){
                Img->label = num;
                field = strtok(NULL, ",");
                iter+=1;
            }
            else{
                Img->grayscale[iter] = num/255.0f;
                field = strtok(NULL, ",");
                iter+=1;
            }
        }
        // printImg(Img->grayscale, 784, 28, Img->label);
        prediction = classifySample(Img, 28);
        // printf("Prediction: %d\n", prediction);
        if (learn){
            backPropogate(Img, learningRate);
        }
        if (prediction == Img->label){
            correct+=1;
        }
        total +=1;
        // printf("Classification accuracy: %f\n",correct/total);
        // sleep(1);
        }
        printf("Classification accuracy: %f\n",correct/total);
        fclose(trainingcsv);
    }
    free(Img->grayscale);
    free(buffer);
    free(Img);
    return;
}
int main(){
    leakyAlpha = 0.01;
    int numEpochs = 50;
    numConvolutionalLayers = 3;
    numFullyConnectedLayers = 4;
    int numFiltersAtConvolutionalLayersArr[4] = {1, 32, 64, 2};
    numFiltersAtConvolutionalLayers = numFiltersAtConvolutionalLayersArr;

    int filterSizesArr[4] = {1, 3, 3, 3};
    filterSizes = filterSizesArr;

    int sizeConvolutionalLayersArr[4] = {784, 784, 196, 49};
    sizeConvolutionalLayers = sizeConvolutionalLayersArr;

    int numNodesAtFullyConnectedLayersArr[5] = {256, 128, 64, 10};
    // int numNodesAtFullyConnectedLayersArr[5] = {128, 64, 32, 10};
    numNodesAtFullyConnectedLayers = numNodesAtFullyConnectedLayersArr; // need to malloc these;

    int filterStridesArr[4] = {0, 2, 2, 1};
    filterStrides = filterStridesArr;
    char* trainingSet = "../MNIST_DATA/mnist_train.csv";
    char* testingSet = "../MNIST_DATA/mnist_test.csv";
    initCNN();
    double learningRate = 0.000090;
    for (int i = 0; i < numEpochs; i++){
        // if (learningRate > 0.001){
        //     learningRate*=0.9;
        // }
        
        printf("Training epoch: %d\n", i);
        printf("Accuracy on training:\n");
        trainCNN(trainingSet, 1,learningRate);
        // sleep(1);
    
        printf("Accuracy on test:\n");
        trainCNN(testingSet, 0,learningRate);
    }
    
    destructCNN();
}

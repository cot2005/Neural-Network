
######################################################################
#functions used to build neural network models
######################################################################

library(neuralnet)
library(NeuralNetTools)
library(randomForest)
library(boot)
library(plyr)
library(ggplot2)
library(pROC)
library(ggbiplot)
library(PRROC)


NN.binary.stkBuildv1.0<-function(processedfile, networkConfig = c(30,25,5),algorithm = "rprop+", 
                          minmax = TRUE, trainProp = 0.9) {
  NNtrainingData <- read.table(processedfile, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  featureData <- as.matrix(NNtrainingData[,-1:-3])
  predCol <- ncol(featureData)   #assumes prediction is last column
  featureData <- na.roughfix(featureData)
  featureNames <- colnames(featureData)
  
  # Scaling data for the NN
  if (minmax == TRUE) {
    scaleObjects <- minMax.scale(featureData)
    scaleData <- scaleObjects$scaleData
  } else {
    scaleObjects <- NULL
    scaleData <- featureData
  }
  # Creates dendrogram for feature assessment.
  NN.dendrogramPlot(scaleData)
  NN.pca(scaleData)
  
  nnFormula <- as.formula(paste(featureNames[predCol], paste(featureNames[-predCol] , collapse = " + "), sep = " ~ "))
  
  # Performs k-fold cross validation (requires unscaled data input then scales in CV)
  # Tuned to be a crude approximation in order to get an AUC curve
  cvResults <- NN.crossValidate.Binary(featureData, nnFormula, networkConfig, predCol, trainPct = trainProp, kfold = 5)
  #makes CV graphs
  NN.ROCplot(cvResults$predDF$Actual, cvResults$predDF$Predicted)
  NN.PRplot(cvResults$predDF, classCol = 1, predCol = 2)
  NN.PPVplot(cvResults$predDF, classCol = 1, predCol = 2, ppvlim = 0.9)
  
  # Down samples full dataset for final model training
  positives <- which(scaleData[,predCol] == 1)
  negatives <- sample(which(scaleData[,predCol] == 0), length(positives))
  trainData <- rbind(scaleData[positives,], scaleData[negatives,])
  
  # Trains the neural network on all the data
  nn <- neuralnet(nnFormula, data = trainData, hidden = networkConfig, linear.output = FALSE, algorithm = "rprop+", 
                  learningrate = 0.01, rep = 10, stepmax = 1e+10, threshold = 0.01)
  
  # Makes Feature importance plot
  NN.importancePlot(nn)
  # Makes the network plot
  pdf("networkPlot.pdf", height = 12, width = 12)
  plotnet(nn, circle_cex = 3, circle_col = "white", bord_col = "black", node_labs = F)
  dev.off()
  
  #saves neural network in a list with the scalars in index 2 and 3 as an RDS
  NNobject <- list(NNmodel = nn, maxs = scaleObjects$maxs, mins = scaleObjects$mins)
  
  nnName <- paste(gsub(".txt", "NNmodel", processedfile))
  networklabel <- paste(networkConfig, collapse = ".")
  rdsName <- paste(nnName, networklabel, ".rds", sep = "")
  saveRDS(NNobject, rdsName)
}

#########Functions for plotting#############

# Dendrogram plotting function. Defaults Euclidean
NN.dendrogramPlot<-function(featureMatrix) {
  euclideanDist <- dist(t(featureMatrix), method = "euclidean")
  euclideanCluster <- hclust(euclideanDist)
  pdf("featureDendrogram.pdf", height = 7, width = 10)
  plot(euclideanCluster)
  dev.off()
}


#expects data with no gene names or IDs as columns
NN.pca<-function(pcaData) {
  predCol <- ncol(pcaData)
  pcaData <- na.omit(pcaData)
  pcaGroups <- data.frame(Features = colnames(pcaData), Colors = c(rep("Metric", (predCol - 1)), "Return"))
  screenmatrix <- pcaData
  screenmatrix <- screenmatrix[is.finite(rowSums(screenmatrix)),]
  screenmatrix <- t(screenmatrix)
  
  pca.input <- screenmatrix[,apply(screenmatrix,2,var,na.rm=TRUE) != 0]
  pca.output <- prcomp(pca.input, center = T, scale. = T) # computes variance
  ggbiplot(pca.output, cex.lab=3, scale = .75, choices = c(1,2), ellipse = F,
           groups = pcaGroups[,2], labels.size = 2,
           var.axes = 'F', label = pcaGroups[,1]) + 
    geom_point(aes(color = pcaGroups[,2]), size=7, pch=19, alpha=0.5) +
    theme(title = element_text(size = 20),axis.title = element_text(size=20),axis.text = element_text(size=20)) +
    ggtitle('Unsupervised PCA', subtitle = "Using stk features from Panel")
  ggsave("PCAnalysis.pdf", width = 8, height = 8)
}

NN.importancePlot<-function(nnModel) {
  importanceData <- olden(nnModel, bar_plot = FALSE)
  importanceData$Features <- row.names(importanceData)
  importanceData <- importanceData[order(importanceData[,1]),]
  importanceData[,2] <- factor(importanceData[,2], levels =importanceData[,2])
  g <- ggplot(importanceData, aes(importanceData[,2], importanceData[,1])) 
  g + geom_bar(stat = "identity", fill = "steelblue") + theme_bw() + xlab("Feature") +
    ylab("Importance") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
  ggsave("oldenImportance.pdf")
}

NN.ROCplot <- function(actualData, predData){
  print(roc)
  pdf("ROCplot.pdf", height = 10, width = 10)
  rocobj <- plot.roc(actualData, predData, ci = TRUE, stratified=FALSE, print.auc=TRUE, show.thres=TRUE, 
                     main = "ROC Curve with CI")
  ciobj <- ci.se(rocobj, specificities = seq(0,1,0.05)) # over a select set of specificities
  plot(ciobj, type = "shape", col = "#CCE5FF")     # plot as a blue shape
  plot(ci(rocobj, of = "thresholds", thresholds = "best"))
  dev.off()
}

NN.PR<-function(datatable, classCol = 1, predCol = 2) {
  datatable[,classCol] <- as.numeric(as.character(datatable[,classCol]))
  pos <- datatable[which(datatable[,classCol] == 1),predCol]
  neg <- datatable[which(datatable[,classCol] == 0),predCol]
  pr <- pr.curve(scores.class0 = pos, scores.class1 = neg, curve = TRUE)
  return(pr)
}

NN.PRplot<-function(datatable, classCol = 1, predCol = 2) {
  pr <- NN.PR(datatable, classCol = 1, predCol = 2)
  pdf("PRplot.pdf", height = 10, width = 10)
  plot(pr)
  dev.off()
}

#function to determine probability threshold for the input PPV limit
NN.PPV<-function(datatable, ppvlim = 0.9, classCol = 1, predCol = 2) {
  sortedDF <- datatable[order(datatable[,predCol], decreasing = T),]
  sortedDF[,classCol] <- as.numeric(as.character(sortedDF[,classCol]))
  sortedDF$PPV <- unlist(lapply(1:nrow(sortedDF), function(x) {sum(sortedDF[1:x,classCol]/x)}))
  PPVthresh <- sortedDF[min(which(sortedDF$PPV < ppvlim)) - 1, predCol]
  ppvdata <- list(data = sortedDF, max = PPVthresh)
  return(ppvdata)
}

NN.PPVplot<-function(datatable,classCol = 1, predCol = 2, ppvlim = 0.9, outputname = "PPVplot.pdf") {
  ppvdata <- NN.PPV(datatable, classCol = 1, predCol = 2, ppvlim = ppvlim)
  pdf(outputname, height = 10, width = 10)
  plot(ppvdata$data[,predCol], ppvdata$data$PPV, xlab = "Prediction Threshold", ylab = "Positive Predictive Value")
  text(0.2, 0.8,  labels = paste("Threshold (", ppvlim, ") = ",signif(ppvdata$max, digits = 3), sep = ""), adj = c(0,0),cex =1.5)
  dev.off()
}


#########Functions for training#############


# Function to optimize network architecture and tuning. Plots 
# variable layer vs mse to help decide the number of layers for network training.
# This function loops the CV by adding the variable layer range to the static layer and
# plotting the resulting changes in MSE and correlation.

NN.optimize.Binary<-function(processedfile, staticLayer = c(5,3), layerRange = c(1:10), kfold = 5, minmax = TRUE, 
                             trainPct = 0.9) {
  NNtrainingData <- read.table(processedfile, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  featureData <- as.matrix(NNtrainingData[,-1:-3])
  predCol <- ncol(featureData)   #assumes prediction is last column
  featureData <- na.roughfix(featureData)
  featureNames <- colnames(featureData)
  
  nnFormula <- as.formula(paste(featureNames[predCol], paste(featureNames[-predCol] , collapse = " + "), sep = " ~ "))
  
  nnMaxPPV <- c()
  nnPRAUC <- c()
  nnAccuracy <- data.frame()
  for (i in layerRange) {
    writeLines("\n")
    print(paste("Testing", i ,"nodes"))
    variableConfig <- c(staticLayer, i)
    cvResults <- NN.crossValidate.Binary(featureData, nnFormula, networkConfig = variableConfig,
                                         predCol = predCol, trainPct = trainPct, kfold = kfold, minmax = minmax)
    ppv <- NN.PPV(cvResults$predDF, ppvlim = 0.9, classCol = 1, predCol = 2)
    predPR <- NN.PR(cvResults$predDF, classCol = 1, predCol = 2)
    
    print(predPR)
    print(paste("Max PPV occurs at ", ppv$max, sep = ""))
    nnMaxPPV <- append(nnMaxPPV, ppv$max)
    nnPRAUC <- append(nnPRAUC, predPR$auc.integral)

    cvAccuracy <- data.frame(Nodes = i, Accuracy = cvResults$accuracyList)
    nnAccuracy <- rbind(nnAccuracy, cvAccuracy)
  }
  ## Performs Graphing
  rangeName <- paste(min(layerRange), max(layerRange), sep = "to")
  staticName <- paste(staticLayer, collapse = ".")
  configName <- paste(staticName, ".",rangeName, ".pdf", sep = "")
  
  # Makes Maximum PPV barplot
  pdf(paste("OptmaxPPVplot",configName, sep = ""), width = 12, height = 7)
  barplot(nnMaxPPV, names.arg = layerRange, main = "CV maxPPVs for NNs",
          xlab = "Nodes", ylab = "Max Threshold")
  dev.off()
  
  # Makes PR barplot
  pdf(paste("OptAURPplot",configName, sep = ""), width = 12, height = 7)
  barplot(nnPRAUC, names.arg = layerRange, main = "CV AUPRCs for NNs",
          xlab = "Nodes", ylab = "AUPRC")
  dev.off()
  
  # Makes MSE barplots
  pdf(paste("OptAccPlot",configName, sep = ""), width = 12, height = 7)
  boxplot(Accuracy~Nodes, data = nnAccuracy, main = "CV Accuracy for NNs", 
          xlab = "Nodes", ylab = "Accuracy")
  dev.off()
}


# K-fold Cross validation function
# requires scaled data matrix. writes a file for CV error and CV test predictions
# and returns a list with the prediction df and error list

NN.crossValidate.Binary<-function(unscaledData, nnFormula, networkConfig = networkConfig, predCol = predCol,
                           trainPct = 0.9, kfold = 10, minmax = TRUE) {
  set.seed(1910)
  cvAccuracy <- c()
  predData <- data.frame()
  if (minmax == TRUE) {
    scaleObjects <- minMax.scale(unscaledData)
    cvData <- scaleObjects$scaleData
  #} else if (minmax == FALSE) {
    #scaleObjects <- zscore.scale(unscaledData[,-predCol])
    #cvData <- cbind(scaleObjects$scaleData, Classification=unscaledData[,predCol])
  } else {
    cvData <- unscaledData
  }
  
  pbar <- create_progress_bar('text')  # Initialize progress bar
  pbar$init(kfold)
  for(i in 1:kfold){
    # downsamples in order to balance the classes
    positives <- which(cvData[,predCol] == 1)
    negatives <- sample(which(cvData[,predCol] == 0), length(positives))
    downsampleDF <- rbind(cvData[positives,], cvData[negatives,])
    
    index <- sample(1:nrow(downsampleDF), round(trainPct * nrow(downsampleDF)))   #takes percent of data randomly
    trainData <- downsampleDF[index,]
    testData <- downsampleDF[-index,]
    
    nn <- neuralnet(nnFormula, data = trainData, hidden = networkConfig, linear.output = FALSE, algorithm = "rprop+", 
                    learningrate = 0.1, rep = 1, stepmax = 1e+10, threshold = 0.01)
    nnPreds <- compute(nn, testData[,-predCol])
    
    # Calculates Accuracy
    nnPredsClass <- nnPreds$net.result
    confusionDF <- table(Actual = testData[,predCol],Predicted = round(nnPredsClass))
    if (length(confusionDF) < 4) {
      nnAccuracy <- 0.5
    } else {
      nnAccuracy <- sum(confusionDF[1,1], confusionDF[2,2])/sum(confusionDF)
    }
    
    cvAccuracy <- append(cvAccuracy, nnAccuracy)
    plotData <- data.frame(Actual = testData[,predCol],Predicted = nnPredsClass)
    predData <- rbind(predData, plotData)
    
    pbar$step()   #updates progress bar
  }   #closes for loop
  write.table(predData, "CVtestsetPredictions.txt", sep = "\t", col.names = T, row.names = F, quote = F)
  writeLines(as.character(cvAccuracy), "CVerror.txt")
  cvData <- list(predDF = predData, accuracyList = cvAccuracy)
  return(cvData)
}




#########Functions for scaling#############

#function for min-max matrix scaling for first time
minMax.scale<-function(dataMatrix) {
  maxs <- unname(apply(dataMatrix, 2, max))
  mins <- unname(apply(dataMatrix, 2, min))
  scaleMatrix <- scale(dataMatrix, center = mins, scale = maxs - mins)
  scaleObjects <- list(scaleData = scaleMatrix, maxs = maxs, mins = mins)
  return(scaleObjects)
}

#function for min-max matrix rescaling other data sets to match a previous scaling
minMax.rescale<-function(dataMatrix, maxs, mins) {
  scaleMatrix <- scale(dataMatrix, center = mins, scale = maxs - mins)
  return(scaleMatrix)
}

#function for min-max list descaling
minMax.descale<-function(predList, maxs, mins) {
  descaledList = predList * (maxs - mins) + mins
  return(descaledList)
}

#function for z score standardization
zscore.scale<-function(dataMatrix) {
  means <- unname(apply(dataMatrix, 2, mean))
  stdDev <- unname(apply(dataMatrix, 2, sd))
  scaleMatrix <- scale(dataMatrix, center = means, scale = stdDev)
  scaleObjects <- list(scaleData = scaleMatrix, means = means, stdDev = stdDev)
  return(scaleObjects)
}

#function for z score list destandardization
zscore.descale<-function(predList, means, stdDev) {
  descaledList = predList * stdDev + means
  return(descaledList)
}


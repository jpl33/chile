require(MASS)
require(tmvtnorm)
require(randomForest)
require(nnet)
require(Fselector)
require(mlr)
require(gridExtra)
require(ggplot2)
require(afex)
require(car)
require(dummies)

dff<-read.csv("clean_chile.csv")

##############################################################################
library(dummies)

#create a dummy data frame
dff_pca<- dummy.data.frame(dff[,-8], names = c("region","sex","education"))

prin_comp <- prcomp(dff_pca, scale. = T)
screeplot(prin_comp,type="lines")
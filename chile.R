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
dff[,"income_fact"]<-cut(dff$income, breaks = 8)
dff[,"population_fact"]<-cut(dff$population, breaks = 10)


##############################################################################
library(dummies)

#create a dummy data frame
dff_pca<- dummy.data.frame(dff[,-c(8,9,10)], names = c("region","sex","education"))
dff_pca2<- dummy.data.frame(dff[,-c(2,6,8)], names = c("region","population_fact","sex","education","income_fact"))


prin_comp <- prcomp(dff_pca, scale. = T)
prin_comp2<- prcomp(dff_pca2, scale. = T)
screeplot(prin_comp,type="lines")
screeplot(prin_comp2,type="lines")

summary(prin_comp)
summary(prin_comp2)



require(MASS)
require(randomForest)
require(nnet)
require(mlr)
require(gridExtra)
require(ggplot2)
require(afex)
require(car)
require(dummies)
require(kernlab)
require(e1071)
require(vegan)
require(xgboost)

dff<-read.csv("clean_chile.csv")



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

##############################################################################

# multinomial logistic regression
lrn_logR<-makeLearner("classif.multinom",predict.type = "prob")
lrn_svm<-makeLearner("classif.randomForest",predict.type = "prob")

task1<-makeClassifTask(data = dff[,-c(9,10)], target = "vote")
task1_norm<-normalizeFeatures(task1, method = "standardize")

# specify 3 subsample iterations, each with 2/3 of data( default), and stratify "region" variable
rdesc = makeResampleDesc("Subsample", iters = 3, stratify.cols = c("region"))

rr<-resample(lrn_logR, task1_norm, rdesc, extract = function(x) x$learner.model$Coefficients, measures = list(mmce,multiclass.aunp),models = TRUE)


library(FSelector)
require(FSelector)

n = getTaskSize(task1)

## Use 2/3 of the observations for training
train.set = sample(n, size = 2*n/3)
test.set<-as.numeric(row.names(dff))[-train.set]

## Train the learner
mod<-train(lrn_logR, task1_norm, subset = train.set)

## predict test results & performance measures
prd<-predict(mod,task1_norm,subset =test.set)


perf<-mlr::performance(prd, measures = list(mmce,multiclass.aunp))



## calculate model z score
afex::set_sum_contrasts() # use sum coding, necessary to make type III LR tests valid

Anova(mod$learner.model,type="III")

#######################################################################################
#  Random Forest

lrn_rf<-makeLearner("classif.rpart",predict.type = "prob")
rr_rf<-resample(lrn_rf, task1, rdesc, extract = function(x) x$learner.model, measures = list(mmce,multiclass.aunp),models = TRUE)

mod_rf<-train(lrn_rf, task1, subset = train.set)
prd_rf<-predict(mod_rf,task1,subset =test.set)
perf_rf<-mlr::performance(prd_rf, measures = list(mmce,multiclass.aunp))


#######################################################################################
#  QDA Forest

lrn_qda<-makeLearner("classif.qda",predict.type = "prob")
rr_qda<-resample(lrn_qda, task1, rdesc, extract = function(x) x$learner.model, measures = list(mmce,multiclass.aunp),models = TRUE)

mod_qda<-train(lrn_qda, task1, subset = train.set)
prd_qda<-predict(mod_qda,task1,subset =test.set)
perf_qda<-mlr::performance(prd_qda, measures = list(mmce,multiclass.aunp))

#######################################################################################
#  XGBoost Forest

lrn_xgb<-makeLearner("classif.xgboost",predict.type = "prob")
##rr_xgb<-resample(lrn_xgb, task1, rdesc, extract = function(x) x$learner.model, measures = list(mmce,multiclass.aunp),models = TRUE)

dmtrx<-xgb.DMatrix(data.matrix(dff[train.set,-c(8:10)]), label = as.numeric(dff[train.set ,"vote"])-1)
dmtrx_tst<-xgb.DMatrix(data.matrix(dff[test.set,-c(8:10)]), label = as.numeric(dff[test.set ,"vote"])-1)
bst <- xgboost(dmtrx, max.depth = 2, eta = 1, nround = 2, nthread = 2, objective = "multi:softprob",num_class=4)
prd_xgb<-predict(bst,data.matrix(dff[test.set,-c(8:10)]),reshape=TRUE)
colnames(prd_xgb)[apply(prd_xgb,1,which.max)]


###########################################################################################


# discrete_ps = makeParamSet(
#   makeDiscreteParam("C", values = c(0.2,0.4 ,0.5, 1.0, 1.5, 2.0)),
#   makeDiscreteParam("sigma", values = c(0.2,0.4,0.5, 1.0, 1.5, 2.0))
# )
# ctrl = makeTuneControlGrid()
# rdesc = makeResampleDesc("CV", iters = 3L)
# res = tuneParams("classif.ksvm", task = task1_norm, resampling = rdesc,
#                  par.set = discrete_ps, control = ctrl)
# 
# 
# svm_model <- svm(vote ~ ., data=dff[,-c(9,10)])
# summary(svm_model)
# pred_svm <- predict(svm_model,data=dff[,-c(8,9,10)])


vote_fact<-model.matrix(~0+dff$vote,dff$vote)
#ad<-adonis(vote_fact~.,data= dff[,-c(8:10)],type="margin")
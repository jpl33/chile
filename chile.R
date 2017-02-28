require(MASS)
require(tmvtnorm)
require(randomForest)
require(nnet)
library(FSelector)
require(Fselector)
require(mlr)
require(gridExtra)
require(ggplot2)
require(afex)
require(car)
require(dummies)
require(kernlab)
require(e1071)

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

rr<-resample(lrn_logR, task1_norm, rdesc, , extract = function(x) x$learner.model$Coefficients, measures = list(mmce,multiclass.aunp),models = TRUE)


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

lrn_rf<-makeLearner("classif.randomForest",predict.type = "prob")
rr_rf<-resample(lrn_rf, task1, rdesc, , extract = function(x) x$learner.model, measures = list(mmce,multiclass.aunp),models = TRUE)

mod_rf<-train(lrn_rf, task1, subset = train.set)
prd_rf<-predict(mod_rf,task1,subset =test.set)
perf_rf<-mlr::performance(prd_rf, measures = list(mmce,multiclass.aunp))


###########################################################################################


discrete_ps = makeParamSet(
  makeDiscreteParam("C", values = c(0.2,0.4 ,0.5, 1.0, 1.5, 2.0)),
  makeDiscreteParam("sigma", values = c(0.2,0.4,0.5, 1.0, 1.5, 2.0))
)
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("CV", iters = 3L)
res = tuneParams("classif.ksvm", task = task1_norm, resampling = rdesc,
                 par.set = discrete_ps, control = ctrl)

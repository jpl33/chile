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

dff<-read.csv("clean_chile.csv")

##############################################################################

# multinomial logistic regression
lrn_logR<-makeLearner("classif.multinom",predict.type = "prob")

task1<-makeClassifTask(data = dff, target = "vote")
task1<-normalizeFeatures(task1, method = "standardize")
# specify 3 subsample iterations, each with 2/3 of data( default), and stratify "region" variable
rdesc = makeResampleDesc("Subsample", iters = 3, stratify.cols = c("region"))

rr = resample(lrn_logR, task1, rdesc, , extract = function(x) x$learner.model$Coefficients, measures = list(mmce,multiclass.aunp),models = TRUE)

n = getTaskSize(task1)
## Use 2/3 of the observations for training
train.set = sample(n, size = 2*n/3)
test.set<-as.numeric(row.names(dff))[-train.set]

## Train the learner
mod<-train(lrn_logR, task1, subset = train.set)
## predict test results & performance measures
prd<-predict(mod,task1,subset =test.set)
perf<-mlr::performance(prd, measures = list(mmce,multiclass.aunp))

thresh<-c(A=198/2700, N=949/2700, U=627/2700, Y=926/2700)
prdt<-setThreshold(prd,threshold = thresh)
perft<-mlr::performance(prdt, measures = list(mmce,multiclass.aunp))
perft
## calculate model z score
afex::set_sum_contrasts() # use sum coding, necessary to make type III LR tests valid

Anova(mod$learner.model,type="III")

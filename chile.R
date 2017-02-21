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

dff<-read.csv("chile.csv")

# old_par<-par()
# par(mfrow=c(3,3))
# for (i in 1:(ncol(dff)-1)){
#   if (i %in% c(1,3,5)){
#     plot(dff[,i],xlab=colnames(dff)[i])
#   }
#   else {
#     hist(dff[,i],xlab=colnames(dff)[i])
#   }
# 
# }
# par(old_par)

# dff[,"statusquo_f"]<-cut(dff[,"statusquo"],breaks = c(-Inf,-1,0,1,Inf),labels = c("neg_lrg","neg_med","pos_med","pos_lrg"))
# dff<-data.frame(dff[1:6],"statusquo"=dff[,9],"vote"=dff[,8])
# plots<-list()
# for (i in 1:(ncol(dff)-1)){
#   plots[[i]]<-qplot(data = data.frame(x = dff[,i] ,y=dff[,"vote"]),x,xlab=colnames(dff)[i])+geom_bar(aes(fill = dff$vote))
# }
# 
# marrangeGrob(plots,ncol=2,nrow=4)

# replace NA values with estimated distributions
dff[which(is.na(dff$age)),"age"]<-36
inc<-fitdistr(dff[!is.na(dff$income),"income"],"normal")
inc_na<-rnorm(nrow(dff[is.na(dff$income),]),inc$estimate,inc$sd)
inc_ind<-as.numeric(row.names(dff[is.na(dff$income),]))

for (i in 1:nrow(dff[is.na(dff$income),])){
  dff[inc_ind[i],"income"]<-inc_na[i]
}

stts<-fitdistr(dff[!is.na(dff$statusquo),"statusquo"],"normal")
stts_na<-rnorm(nrow(dff[is.na(dff$statusquo),]),stts$estimate,stts$sd)
stts_ind<-as.numeric(row.names(dff[is.na(dff$statusquo),]))

for (i in 1:nrow(dff[is.na(dff$statusquo),])){
  dff[stts_ind[i],"statusquo"]<-stts_na[i]
}

ind<-as.numeric(row.names(dff[is.na(dff$education),]))
for (i in 1:nrow((dff[is.na(dff$education),]))){
  if(i<2){
    dff[ind[i],"education"]<-"PS"
  }
  else if(i<5){
    dff[ind[i],"education"]<-"S"
  }
  else{
    dff[ind[i],"education"]<-"P"
  }
}

ind<-as.numeric(row.names(dff[is.na(dff$vote),]))
for (i in 1:nrow((dff[is.na(dff$vote),]))){
  if(i<12){
    dff[ind[i],"vote"]<-"A"
  }
  else if(i<51){
    dff[ind[i],"vote"]<-"U"
  }
  else if(i<109){
    dff[ind[i],"vote"]<-"Y"
  }
  else {
    dff[ind[i],"vote"]<-"N"
  }
}
##############################################################################

# multinomial logistic regression
lrn_logR<-makeLearner("classif.multinom",predict.type = "prob")

task1<-makeClassifTask(data = dff, target = "vote")
# specify 3 subsample iterations, each with 2/3 of data( default), and stratify "region" variable
rdesc = makeResampleDesc("Subsample", iters = 3, stratify.cols = c("region"))

rr = resample(lrn_logR, task1, rdesc, measures = list(mmce,multiclass.aunp),models = TRUE)
getConfMatrix(rr$pred)

n = getTaskSize(task1)
## Use 2/3 of the observations for training
train.set = sample(n, size = 2*n/3)
test.set<-as.numeric(row.names(dff))[-train.set]

## Train the learner
mod<-train(lrn_logR, task1, subset = train.set)
## predict test results & performance measures
prd<-predict(mod,task1,subset =test.set)
perf<-mlr::performance(prd, measures = list(mmce,multiclass.aunp))

## calculate model z score
afex::set_sum_contrasts() # use sum coding, necessary to make type III LR tests valid

Anova(mod$learner.model,type="III")

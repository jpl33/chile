require(MASS)
require(tmvtnorm)
require(randomForest)
require(nnet)
require(mlr)
require(gridExtra)
require(ggplot2)
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


# preparing CV-3 data split
k<-3
# first we split the data frame by  the "region" factor
#df_lst<-split(dff,dff$vote)
df_lst<-split(dff,dff$region)
# then, we split each homogenous "region" data frame into three 
dfi_folds<-list()
for (i in 1:k){
  dfi_folds[[i]] <- cut(seq(1,nrow(df_lst[[i]])),breaks=k,labels=FALSE)
}
# then, we recombine all "vote" thirds into three, heterogenous "vote", data frames 
data_cv3<-list()
for (i in 1:k){
  t<-data.frame(rbind(df_lst[[1]][which(dfi_folds[[1]]==i),],df_lst[[2]][which(dfi_folds[[2]]==i),],df_lst[[3]][which(dfi_folds[[3]]==i),]))
  levels(t$region)<-levels(dff$region)
  data_cv3[[i]]<-t
  }


# multinomial logistic regression
lrn_logR<-makeLearner("classif.multinom",predict.type = "prob")
train<-list()
predict<-list()
for (j in k:1){
  task_orig<-makeClassifTask(data = data_cv3[[j]], target = "vote")
  train[[j]]<-train(lrn_logR,task_orig)
  assign(paste("train_",j,sep=""),train[[j]])
  t<-data.frame(ifelse(j>2,data_cv3[1],data_cv3[j+1]))
  assign(paste("predict_",j,sep=""),predict(train[[j]],newdata =t ))
  
}

mlr::performance(predict_1)
mlr::performance(predict_2,measures=list(multiclass.au1p,mmce))
getConfMatrix(predict_1)


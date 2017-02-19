require(MASS)
require(tmvtnorm)
require(randomForest)
require(nnet)
require(mlr)
require(gridExtra)
require(ggplot2)
require(ROCR)
require(DMwR)
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


# preparing CV-3 data split
k<-3
# first we split the data frame by  the "vote" factor
df_lst<-split(dff,dff$vote)

# then, we split each homogenous "vote" data frame into three 
dfi_folds<-list()
for (i in 1:length(levels(dff$vote))){
  dfi_folds[[i]] <- cut(seq(1,nrow(df_lst[[i]])),breaks=k,labels=FALSE)
}
# then, we recombine all "vote" thirds into three heterogenous "vote" data frames 
data_cv3<-list()
for (i in 1:k){
  t<-data.frame(rbind(df_lst[[1]][which(dfi_folds[[1]]==i),],df_lst[[2]][which(dfi_folds[[2]]==i),],df_lst[[3]][which(dfi_folds[[3]]==i),],df_lst[[4]][which(dfi_folds[[4]]==i),]))
  data_cv3[[i]]<-t
}

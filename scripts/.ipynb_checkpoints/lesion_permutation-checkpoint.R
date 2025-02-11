####
# CLEAR AND LOAD WORKSPACE
rm(list = ls())
gc()

# Separator
sep <- "/"

# Get the current working directory
current_directory <- getwd()

# Derive the repository path (parent directory with a trailing separator)
repo_path <- paste0(paste(head(strsplit(current_directory, sep)[[1]], -1), collapse = sep), sep)

# Construct the subdirectory paths
notebook_path <- paste0(repo_path, "notebook", sep)
input_path    <- paste0(repo_path, "input", sep)
output_path   <- paste0(repo_path, "output", sep)
scripts_path  <- paste0(repo_path, "scripts", sep)
imgs_path     <- paste0(repo_path, "imgs", sep)

load(paste0(input_path,'perm_lesions.RData'))

#### 
# LIBRARIES AND PATH
library(foreach)
library(doFuture)
library(utils)
library(scales)
library(oro.nifti)
library(MNITemplate)
library(neurobase)
library(reticulate)

####
# CORRECT P-VALUES
maxTlightlivio <- function (permT, alphas = c(.001,.01,.05,.1), weights = NULL, m = ncol(permT)) 
{
  if (is.null(colnames(permT))) 
    colnames(permT) = 1:m

  alphas=ceiling(alphas*nrow(permT))/nrow(permT)
  alphas=sort(alphas)
  
  if (!is.null(weights)) 
    permT = t(weights * t(permT))
  
  Padjs = rep(1, m)
  names(Padjs) = colnames(permT)
  
  alpha=alphas[1]
  nalphas=length(alphas)
  
  alphaid=1
  
  Padjs=rep(1,m)
  notrejs=1:m
  
  while ((alphaid <= nalphas)&(length(notrejs)>0)) 
  {
    th=compute_thresholds(permT[,notrejs,drop=FALSE],alphas[alphaid])
    tmp=which(permT[1,notrejs]>th)

    while(length(tmp)>0){
      Padjs[notrejs[tmp]]=alphas[alphaid]
      notrejs=notrejs[-tmp]
      included_in_max=notrejs
      if(length(notrejs)==0)return(Padjs)
      th=compute_thresholds(permT[,notrejs,drop=FALSE],alphas[alphaid])
      tmp=which(permT[1,notrejs]>th)
    } 
    alphaid=alphaid+1
  }
  return(Padjs)
}

compute_thresholds <- function(permT,alphas){
  quantile(do.call(pmax, permT),round(1-alphas,10),type=1,names=FALSE)
}

#### 
# DECIDE WHICH CLUSTERS TO COMPARE
i = 4

####
# SET HYPERPARAMETERS
N_perm = 1000 - 1 #In the article it was used 5000
ncores = 10
alphas = c(0.001,0.005,0.01)

#### 
# EXTRACT MIN NUMBER OF LESIONS x VOXEL
vth = 8

####
print('Extracting subjects from lesion file...')
start <- Sys.time()

# EXTRACT PATIENTS x VOXEL MATRIX
eval(parse(text=paste0('maskC', i,' = (labels==',i,')')))
eval(parse(text=paste0('maskC = (labels!=',i,')')))

tmask = (colSums(lesions)>=vth)

tindexmask = which(tmask)
v_os = data.matrix(lesions[,tmask])

rm(lesions)

labelsij = labels
print(Sys.time() - start)

####
# CALCULATE ORIGINAL STATISTIC
print('Calculating Original Statistic...')
start <- Sys.time()

eval(parse(text=paste0('C',i,'=sum(maskC',i,')')))
eval(parse(text=paste0('Ci=sum(maskC',i,')')))

eval(parse(text=paste0('C=sum(maskC)')))

N_voxel = dim(v_os)[2]
print(N_voxel)
es_or = numeric(N_voxel)

for (v in 1:N_voxel)
{
  k  = v_os[, v]
  
  #DIFFERENCES IN PROPORTION
  si = sum(k[labelsij==i])
  st = sum(k)
  pi = si/Ci
  pj = (st-si)/C
  p  = st/(Ci+C)
  
  # sqrt-transform
  #pi = sqrt(pi + 0.5)
  #pj = sqrt(pj + 0.5)
  
  # continuity correction
  c_corr = 0
  #c_corr = +(1/Ci + 1/C)/2
  
  #double side
  #pooled
  es_or[v] = (abs(pi-pj) - c_corr)/sqrt(p*(1-p)*(1/Ci + 1/C)+.Machine$double.eps)
  #unpooled
  #es_or[v] = (abs(pi-pj) - c_corr)/sqrt(pi*(1-pi)/Ci + pj*(1-pj)/C +.Machine$double.eps)
  
  #one side
  #pooled
  #es_or[v] = (pi-pj -c_corr)/sqrt(p*(1-p)*(1/Ci + 1/C)+.Machine$double.eps)
  #unpooled
  #es_or[v] = (pi-pj + c_corr)/sqrt(pi*(1-pi)/Ci + pj*(1-pj)/C +.Machine$double.eps)

}
print(Sys.time() - start)

####
# CALCULATE PERMUTATION STATISTIC
print('Calculating Permutations...')
start <- Sys.time()

set.seed(12345)
index_perm = replicate(N_perm,sample(Ci+C))

plan(multisession, workers = ncores)

es_perm <- foreach(x = 1:N_perm, .combine = 'rbind') %dofuture% {
  index_perm_x = index_perm[,x]
  vapply(1:N_voxel, 

         # DIFFERENCES IN PROPORTION
         function(v){
          
           k  = v_os[index_perm_x, v]
           si = sum(k[labelsij==i])
           st = sum(k)
           
           pi = si/Ci
           pj = (st-si)/C
           p  = st/(Ci+C)
           
           # sqrt-transformed
           #pi = sqrt(pi + 0.5)
           #pj = sqrt(pj + 0.5)
           
           # continuity correction
           c_corr = 0
           #c_corr = +(1/Ci + 1/C)/2
           
           #double-side
           #pooled
           (abs(pi-pj)-c_corr)/sqrt(p*(1-p)*(1/Ci + 1/C)+.Machine$double.eps)
           #unpooled
           #(abs(pi-pj)-c_corr)/sqrt(pi*(1-pi)/Ci + pj*(1-pj)/C +.Machine$double.eps)
           
           #one-side
           #pooled
           #(pi-pj - c_corr)/sqrt(p*(1-p)*(1/Ci + 1/C)+.Machine$double.eps)
           #unpooled
           #(pi-pj +c_corr)/sqrt(pi*(1-pi)/Ci + pj*(1-pj)/C +.Machine$double.eps)
           
         },
         numeric(1))
}

plan(sequential)
print(Sys.time() - start)

####
# CALCULATE ADJUSTED P VALUES
print('Adjusting p-values...')
start <- Sys.time()

stat_orandperm = as.data.frame(rbind(es_or,es_perm))
rm(es_perm)

adj_p_val = maxTlightlivio(stat_orandperm, alphas = alphas)
smask = (adj_p_val!=1)
sign_stat = sign(es_or[smask])
sindexmask = tindexmask[smask]
N_voxel_s = sum(smask)
print(N_voxel_s)
print(Sys.time() - start)

####
# SET FINAL MASK FOR IMAGING
tmask[tindexmask] = FALSE
tmask[sindexmask] = TRUE

#### 
# PLOT SIGNIFICANT VOXEL
img = readMNI("Brain")
if (sum(tmask)>0){
  mpt = array_reshape(tmask,dim=c(182,218,182),order="C")
  maskp = img
  maskp[mpt] <- 1
  maskp[!mpt] <- NA
  ortho2(img, maskp,c(120,80,80), col.y = alpha("red",0.3), add.orient=TRUE)
}

#### 
# WRITE LESION FILE
write.csv(tmask,file=paste0(output_path,'C',i,'vsC_smask.csv'))

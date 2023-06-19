// #include <R.h>

void generate_codebook(int *nodestatus, int *nofnode, int *noftree, int *terminal,  int *nofterminal, int *nofobservations, int *total, int *nofseries, double *result)
{
     int i,j,k,index,temp,ind;
     double tmp;
     for(i=0;i<*noftree;i++){
		temp=0;
		for(j=0;j<*nofnode;j++){
			if(nodestatus[*nofnode*i+j]<0){
				nodestatus[*nofnode*i+j]=temp;
				temp++;			
			}
		}
		temp=0;
		for(k=0;k<*nofseries;k++){
			for(j=0;j<*nofterminal;j++){
				result[(*nofseries)*(i*(*nofterminal)+j)+k]=0;
			}
			tmp=nofobservations[k];
			for(j=0;j<nofobservations[k];j++){
				ind=terminal[(*total*i)+temp+j]-1;
				index=nodestatus[i*(*nofnode)+ind];
				result[(*nofseries)*(i*(*nofterminal)+index)+k]=result[(*nofseries)*(i*(*nofterminal)+index)+k]+1/tmp;
			}			
			temp=temp+nofobservations[k];
		}
     }
}

import numpy as np


# Tune hyper-parameters here.
#opts = {
#    'threshold': 100000000,
#    'num_epochs': 100000000,
#    'batch_size': 100000000,
#    'init_weight_scale': 10000000000.000000000000001,
#    'learning_rate': 10000000000.000000000000001
#}
opts = {
    'threshold':10,
    'num_epochs': 100,
    'batch_size': 20,
    'init_weight_scale': 0.02,
    'learning_rate': 0.6
}

class LinearLayerForward:
    def __call__(self, weights, xs, ctx=None):
        """
        Implement a batched version of linear transformation.
        """
        s = xs.shape[0]
        ls_logits = []
        for i in range(s):
            l_i = np.dot(weights,xs[i])
            ls_logits.append(l_i)
            
        logits = np.array(ls_logits)
        if ctx is not None:
            for i in range(s):
                ctx[i] = xs[i] 

        return logits


class LinearLayerBackward:
    def __call__(self, ctx, dlogits):
        """
        Get the derivative of the weight vector.
        """
        s = len(ctx)
        f = len(ctx[0])
        dw = np.zeros(f)
        for i in range(s):
            dw = dw+ dlogits[i]*ctx[i]
#        ls_dw = []
#        for t in range(f):
#            g_t = 0 
#            for i in range(s):
#                g_t += dlogits[i]*ctx[i][t]
#            ls_dw.append(g_t)
#        dw = np.array(ls_dw)
                
        return dw


class LinearLayerUpdate:
    def __call__(self, weights, dw, learning_rate=1.0):
        """
        Update the weight vector.
        """

        new_weights = weights - learning_rate*dw

        return new_weights


class SigmoidCrossEntropyForward:
    def __call__(self, logits, ys, ctx=None):
        """
        Implement a batched version of sigmoid cross entropy function.
        """
        
        s = ys.shape[0]
        loss = 0
        for i in range(s):
            if logits[i]<0:
                loss += np.log(np.e**logits[i]+1) -ys[i]*logits[i]
            else:
                loss += logits[i]*(1-ys[i]) + np.log(np.e**(-logits[i])+1)
                
        average_loss = loss/s
        
        if ctx is not None:
            # Put your code here.
            for i in range(s):
                ctx[i] = [ys[i],logits[i]]

        return average_loss


class SigmoidCrossEntropyBackward:
    def __call__(self, ctx, dloss):
        """
        Get the derivative of logits.
        """
        ls_dlogits = []
        s = len(ctx)
        for i in range(s):
            if ctx[i][1]<0:
#                dlogits_i = -ctx[i][0]+1/(1+np.e**(-ctx[i][1]))
                dlogits_i =np.e**(ctx[i][1])/(1+np.e**(ctx[i][1])) -ctx[i][0]
                ls_dlogits.append(dlogits_i)
            else:
                dlogits_i = 1 -ctx[i][0]-np.e**(-ctx[i][1])/(1+np.e**(-ctx[i][1]))
                ls_dlogits.append(dlogits_i)
#                try:
#                    dlogits_i = 1 -ctx[i][0]-1/(1+np.e**(ctx[i][1]))
#                    ls_dlogits.append(dlogits_i)
#                except:
#    #                dlogits_i = 1 -ctx[i][0]-1/(1+np.e**(ctx[i][1]))
#                    dlogits_i =np.e**(ctx[i][1])/(1+np.e**(ctx[i][1])) -ctx[i][0]
#                    ls_dlogits.append(dlogits_i)
        dlogits = np.array(ls_dlogits)/s
        return dlogits


class Prediction:
    def __call__(self, logits):
        """
        Make email classification.
        """

        s = logits.shape[0]
        ls_prediction = []
        for i in range(s):
            if logits[i]>=0:
                ls_prediction.append(True)
            else:
                ls_prediction.append(False)
        predictions = np.array(ls_prediction)

        return predictions

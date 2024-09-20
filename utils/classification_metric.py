import torch
import torchmetrics


class Classification(torchmetrics.Metric):
    def __init__(self, num_classes, average=None, conf_mat_multilabel=False,evaluate_conf_matrix=True, task='multiclass',verbose=False):
        super().__init__()

        conf_mat_normalize='none' 
        self.evaluate_conf_matrix = evaluate_conf_matrix

        # Give default behaviour macro for multiclass 
        if average is None and verbose==True:
            if num_classes >=2:
                average='macro'
                print ("Average behaviour is set to {}, see: https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html for details".format(average))
            else:
                average='micro'
                print ("Average behaviour is set to {}, see: https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html for details".format(average))


        self.metric_acc         = torchmetrics.Accuracy(task=task,num_classes=num_classes,average=average, mdmc_average='global')
        self.metric_mcc         = torchmetrics.MatthewsCorrCoef(task=task,num_classes=num_classes)
        self.metric_kappa       = torchmetrics.CohenKappa(task=task,num_classes=num_classes)

        self.metric_prec        = torchmetrics.Precision(task=task,num_classes=num_classes,average=average, mdmc_average='global')
        self.metric_recall      = torchmetrics.Recall(task=task,num_classes=num_classes,average=average, mdmc_average='global')
        if self.evaluate_conf_matrix == True:
            self.metric_conf_mat    =  torchmetrics.ConfusionMatrix(task=task,num_classes=num_classes, normalize=conf_mat_normalize, multilabel=conf_mat_multilabel)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.metric_acc(preds,target)
        self.metric_mcc(preds,target) 
        self.metric_kappa(preds,target)
        self.metric_prec(preds,target)
        self.metric_recall(preds,target)
        self.metric_conf_mat(preds,target)


    def compute(self):

        acc         = self.metric_acc.compute()
        mcc         = self.metric_mcc.compute()
        kappa       = self.metric_kappa.compute()
        precision   = self.metric_prec.compute()
        recall      = self.metric_recall.compute()
        

        if self.evaluate_conf_matrix == True:
            conf_mat    = self.metric_conf_mat.compute()
            return {'acc':acc, 'mcc':mcc, 'kappa':kappa, 'precision':precision, 'recall':recall, 'conf_mat':conf_mat}

        else:
            return {'acc':acc, 'kappa':kappa, 'precision':precision, 'recall':recall}




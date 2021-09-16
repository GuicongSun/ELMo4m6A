import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_curve, roc_auc_score, auc


class map():
    # Calculate the performance of each evaluation indicator
    def calculateScore(X, y, model):
        score = model.evaluate(X, y)

        accuracy = score[1]

        pred_y = model.predict(X)

        tempLabel = np.zeros(shape=y.shape, dtype=np.int32)

        for i in range(len(y)):
            if pred_y[i] < 0.4:
                tempLabel[i] = 0
            else:
                tempLabel[i] = 1

        confusion = confusion_matrix(y, tempLabel)
        TN, FP, FN, TP = confusion.ravel()

        sensitivity = recall_score(y, tempLabel)
        specificity = TN / float(TN + FP)
        MCC = matthews_corrcoef(y, tempLabel)

        F1Score = (2 * TP) / float(2 * TP + FP + FN)
        precision = TP / float(TP + FP)

        pred_y = pred_y.reshape((-1,))

        # Calculate the area under the ROC curve (AUC).
        ROCArea = roc_auc_score(y, pred_y)
        # fpr and tpr are used to draw the ROC curve.
        fpr, tpr, thresholds = roc_curve(y, pred_y)

        return {'sn': sensitivity, 'sp': specificity, 'acc': accuracy, 'MCC': MCC, 'AUC': ROCArea,
                'precision': precision,
                'F1': F1Score, 'fpr': fpr, 'tpr': tpr}

    # Results of benchmark dataset validation
    def analyze(temp, OutputDir):
        validation_result, testing_result = temp

        file = open(OutputDir + '/performance.txt', 'w')

        index = 0
        for x in [validation_result, testing_result]:

            title = ''

            if index == 0:
                title = 'validation_'
            if index == 1:
                title = 'testing_'

            index += 1

            file.write(title + 'results\n')

            for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1']:

                total = []

                for val in x:
                    total.append(val[j])
                # Calculate and print the mean and standard deviation
                file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total)) + '\n')

            file.write('\n\n______________________________\n')
        file.close()

        # Draw ROC curve
        index = 0
        for x in [validation_result, testing_result]:

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            i = 0
            for val in x:
                tpr = val['tpr']
                fpr = val['fpr']
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                #   numpy.interp()  The main use scenario is one-dimensional linear interpolation
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i + 1, roc_auc))

                i += 1
            # print()

            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
            # Diagonal line
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0

            # if index == 1:
            #     dataframe = pd.DataFrame({'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr})
            #     dataframe.to_csv(OutputDir + "/auc_validation.csv", index=False, sep=',')

            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.4f)' % (mean_auc),
                     lw=2, alpha=.8)
            # Plot standard deviation area
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)  # Select upper bound
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)  # Select lower bound
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")

            title = ''

            if index == 0:
                title = 'validation_'
            if index == 1:
                title = 'testing_'

            plt.savefig(OutputDir + '/' + title + 'ROC.png')
            plt.close('all')

            index += 1


import numpy as np
'''
    True
      P   N
  P  [TP, FP]
  N  [FN, TN]
'''

'''INPUT'''

labels = ['rain', 'norain']

cf_matrix = np.array(
    [[1000,200],
     [800,800]]
)

requested = 0 # = pos, 0 = all

'''----------------------'''


def precision(class_of_interest, confusion_matrix_test):
    tp = confusion_matrix_test[class_of_interest][class_of_interest]
    return tp/sum(confusion_matrix_test[class_of_interest, :])


def recall(class_of_interest, confusion_matrix_test):
    tp = confusion_matrix_test[class_of_interest][class_of_interest]
    return tp/sum(confusion_matrix_test[:, class_of_interest])


def f1_score(precision_value, recall_value):
    return 2*precision_value*recall_value/(precision_value+recall_value)

end = range(0, len(labels)) if requested == 0 else [requested]


for i in end:
    precision_value = precision(i, cf_matrix)
    recall_value = recall(i, cf_matrix)
    f1_score_test = f1_score(precision_value, recall_value)

    print(f"{labels[i]} : Precision: {precision_value}")
    print(f"{labels[i]} : Recall: {recall_value}")
    print(f"{labels[i]} : F1-Score: {f1_score_test}")
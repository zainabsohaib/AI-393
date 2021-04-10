

y_pred_bnb = bnb.fit(X_train, y_train).predict(X_test)
mnb = MultinomialNB()
y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)

accuracy_gnb = metrics.accuracy_score(y_test, y_pred_gnb)
accuracy_bnb = metrics.accuracy_score(y_test, y_pred_bnb)
accuracy_mnb = metrics.accuracy_score(y_test, y_pred_mnb)
print("Accuracy of GaussianNB: ",accuracy_gnb," Accuracy of BernoulliNB: ",accuracy_bnb," Accuracy of MultinomialNB: ",accuracy_mnb)

svm_clf = SVC(kernel="rbf", random_state=42, verbose=3,C=9)
svm_clf.fit(X_train, y_train)

y_test_pred_svm = svm_clf.predict(X_test)

metrics.accuracy_score(y_test, y_test_pred_svm)

test=pd.read_csv("../input/digit-recognizer/test.csv")
test=test/255
svmFinalpred=svm_clf.predict(test)

finalPred=pd.DataFrame(svmFinalpred,columns=["Label"])

finalPred['ImageId']=finalPred.index+1
finalPred = finalPred.reindex(['ImageId','Label'], axis=1)

finalPred.to_csv('./submition.csv',index=False)

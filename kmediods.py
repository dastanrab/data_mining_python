import numpy as np
from sklearn import datasets, decomposition, cluster
import ml_helpers
import sys

class KMediods():
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations
        self.kmediods_centroids = []

    # از مراکز داده داده شده ، به صورت تصادفی به centroid انتخاب کنید
    def _init_random_centroids(self, data):
        n_samples, n_features = np.shape(data)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = data[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # فهرست نزدیکترین مرکز centeroid را به نمونه برگردانید
    def _closest_centroid(self, sample, centroids):
        closest_i = None
        closest_distance = float("inf")
        for i, centroid in enumerate(centroids):
            distance = ml_helpers.euclidean_distance(sample, centroid)
            if distance < closest_distance:
                closest_i = i
                closest_distance = distance
        return closest_i

    #برای ایجاد خوشه ، نمونه ها را به نزدیکترین سانتروئیدها اختصاص دهید
    def _create_clusters(self, centroids, data):
        n_samples = np.shape(data)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(data):		
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # centroids جدید را به عنوان وسیله نمونه در هر خوشه محاسبه کنید
    def _calculate_centroids(self, clusters, data):
        n_features = np.shape(data)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
        	curr_cluster = data[cluster]
        	smallest_dist = float("inf")
        	for point in curr_cluster:
        		total_dist = np.sum(ml_helpers.euclidean_distance(curr_cluster, [point] * len(curr_cluster)))
        		if total_dist < smallest_dist:
        			centroids[i] = point
        return centroids

    # نمونه ها را به عنوان شاخص خوشه های آنها طبقه بندی کنید
    def _get_cluster_labels(self, clusters, data):
        # برای هر نمونه یک پیش بینی
        y_pred = np.zeros(np.shape(data)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # را خوشه بندی کنید و سانتروئیدهای خوشه ها را برگردانیدK-Medoids
    def fit(self, data):
        # شروع سانتروئیدها
        centroids = self._init_random_centroids(data)

        # تکرار تا همگرایی یا برای حداکثر تکرار
        for _ in range(self.max_iterations):
            # اختصاص نمونه ها به نزدیکترین سانتروئیدها (ایجاد خوشه)
            clusters = self._create_clusters(centroids, data)

            prev_centroids = centroids
            # محاسبه centroids جدید از خوشه ها
            centroids = self._calculate_centroids(clusters, data)

            # اگر هیچ سانتروید تغییر نکرده باشد => همگرایی
            diff = centroids - prev_centroids
            if not diff.any():
                break

        self.kmediods_centroids = centroids
        return clusters


    # کلاس هر نمونه را پیش بینی کنید
    def predict(self, data):
        # ابتدا بررسی کنید آیا سنتروئیدهای K-Medoids را تعیین کرده ایم
        if not self.kmediods_centroids.any():
            raise Exception("Mean-Shift centroids have not yet been determined.\nRun the Mean-Shift 'fit' function first.")

        predicted_labels = np.zeros(len(data))
        for i in range(len(predicted_labels)):
        	curr_sample = data[i]
        	distances = [np.linalg.norm(curr_sample - centroid) for centroid in self.kmediods_centroids]
        	label = (distances.index(min(distances)))
        	predicted_labels[i] = label
        	
        return predicted_labels


# اطلاعات آموزش را دریافت کنید
# انتقال فایل داده های iris برای استفاده به عنوان داده اموزشی
iris = datasets.load_iris()
train_data = np.array(iris.data)
train_labels = np.array(iris.target)
num_features = train_data.data.shape[1]

# PCA را برای کاهش ابعاد داده اعمال کنید
pca = decomposition.PCA(n_components=3)
pca.fit(train_data)
train_data = pca.transform(train_data)

# *********************************************
# Apply K-Mediods Clustering MANUALLY
# *********************************************
#اشیا را برای
# خوشه K-Medoids را ایجاد کنید
unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)
clf = KMediods(k=num_classes, max_iterations=3000)

centroids = clf.fit(train_data)

predicted_labels = clf.predict(train_data)


# محاسبه دقت
Accuracy = 0
for index in range(len(train_labels)):
	#خوشه بندی با  kmedios
	current_label = train_labels[index]
	predicted_label = predicted_labels[index]

	if current_label == predicted_label:
		Accuracy += 1

Accuracy /= len(train_labels)


print("دقت طبقه بندی دستی K-Medoids = ", Accuracy)
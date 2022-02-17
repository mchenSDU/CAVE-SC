import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;


public class CAVESC {

	int EXP_TABLE_SIZE = 1000;
	double[] expTable = new double[EXP_TABLE_SIZE];
	int MAX_EXP = 6;

	private double alpha = 0.01;
	double lambda = 0.001;

	public static void main(String[] args) throws IOException {

		CAVESC cm = new CAVESC();
		String[] city = { "TKY", "NYC" };
		String basePath = "data\\";
		for (int m = 0; m < city.length; m++) {
			String resultPath = basePath + city[m] + "\\MRRresult.csv";
			PrintWriter pw = new PrintWriter(new FileWriter(resultPath));

			for (int sampleRate = 10; sampleRate <= 80; sampleRate += 10) {
				String trainingPath = basePath + city[m] + "\\trajectoryTraining" + sampleRate + ".txt";
				ArrayList<String> trainList = new ArrayList<String>();
				int contextWindowSize = 5;
				cm.getTrainList(trainingPath, trainList, contextWindowSize);

				ArrayList<String> poiFrequencyList = new ArrayList<String>();
				ArrayList<String> categoryFrequencyList = new ArrayList<String>();
				ArrayList<String> categoryList = new ArrayList<String>();
				ArrayList<String> poiList = new ArrayList<String>();
				cm.computeTargetFrequencyList(trainList, poiFrequencyList, categoryFrequencyList, categoryList,
						poiList);

				int[] negativeNum = { 1, 2, 3, 4, 5, 6, 7, 8 };
				int[] negativeCategoryNum = { 1, 2, 3, 4, 5, 6, 7, 8 };
				int vecSize[] = { 50, 100, 150, 200, 250, 300 };
				double[] weight = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
				for (int i = 0; i < negativeNum.length; i++) {
					for (int j = 0; j < vecSize.length; j++) {
						for (int k = 0; k < weight.length; k++) {
							for (int h = 0; h < negativeCategoryNum.length; h++) {

								HashMap<String, double[]> contextPOIVecMap = new HashMap<String, double[]>();
								HashMap<String, double[]> targetPOIVecMap = new HashMap<String, double[]>();
								HashMap<String, double[]> contextCategoryVecMap = new HashMap<String, double[]>();
								cm.initialize(vecSize[j], contextPOIVecMap, targetPOIVecMap, contextCategoryVecMap,
										poiList, categoryList);
								System.out.println(negativeNum[i] + "-" + negativeCategoryNum[h] + "-" + vecSize[j]
										+ "-" + weight[k]);
								cm.trainModelWeight(trainList, contextPOIVecMap, targetPOIVecMap, contextCategoryVecMap,
										poiFrequencyList, categoryFrequencyList, categoryList, negativeNum[i],
										weight[k], negativeCategoryNum[h]);
								System.out.println("trainComplete");

								String contextPOIPath = basePath + city[m] + "\\embedding\\contextPOI" + negativeNum[i]
										+ "-" + negativeCategoryNum[h] + "-" + vecSize[j] + "-" + weight[k] + "-"
										+ sampleRate + ".csv";
								String targetPOIPath = basePath + city[m] + "\\embedding\\targetPOI" + negativeNum[i]
										+ "-" + negativeCategoryNum[h] + "-" + vecSize[j] + "-" + weight[k] + "-"
										+ sampleRate + ".csv";
								String contextCategoryPath = basePath + city[m] + "\\embedding\\contextCategory"
										+ negativeNum[i] + "-" + negativeCategoryNum[h] + "-" + vecSize[j] + "-"
										+ weight[k] + "-" + sampleRate + ".csv";
								cm.saveModel(contextPOIVecMap, contextPOIPath);
								cm.saveModel(targetPOIVecMap, targetPOIPath);
								cm.saveModel(contextCategoryVecMap, contextCategoryPath);

//								MRR test = new MRR();
//								String testPath = basePath + city[m] + "\\testPOI" + sampleRate + ".txt";
//								double mrr = test.computeMRRBasedCosine(testPath, contextPOIVecMap,
//										contextCategoryVecMap);
//
//								pw.println(negativeNum[i] + "\t" + negativeCategoryNum[h] + "\t" + vecSize[j] + "\t"
//										+ weight[k] + "\t" + sampleRate + "\t" + mrr);
							}
						}
					}
				}
			}
			pw.flush();
			pw.close();
		}

	}

	private void initialize(int vecSize, HashMap<String, double[]> contextPOIVecMap,
			HashMap<String, double[]> targetPOIVecMap, HashMap<String, double[]> contextCategoryVecMap,
			ArrayList<String> poiList, ArrayList<String> categoryList) throws IOException {
		// sigmoid function
		createExpTable();

		for (int i = 0; i < categoryList.size(); i++) {
			String category = categoryList.get(i);
			double[] vec1 = new double[vecSize];
			Random random = new Random();
			for (int j = 0; j < vec1.length; j++) {
				vec1[j] = random.nextGaussian() * 0.01;
			}
			contextCategoryVecMap.put(category, vec1);
		}

		for (int i = 0; i < poiList.size(); i++) {
			String poi = poiList.get(i);
			double[] vec1 = new double[vecSize];
			Random random = new Random();
			for (int j = 0; j < vec1.length; j++) {
				vec1[j] = random.nextGaussian() * 0.01;
			}
			contextPOIVecMap.put(poi, vec1);
			targetPOIVecMap.put(poi, vec1);
		}
	}

	private void trainModelWeight(ArrayList<String> trainList, HashMap<String, double[]> contextPOIVecMap,
			HashMap<String, double[]> targetPOIVecMap, HashMap<String, double[]> contextCategoryVecMap,
			ArrayList<String> targetPOIFrequencyList, ArrayList<String> targetCategoryFrequencyList,
			ArrayList<String> categoryList, int negativeNum, double weight, int negativeCategoryNum)
			throws IOException {
		int itrNum = 0;
		while (true) {
			System.out.println("The " + (itrNum + 1) + "th iteration starts.");
			Collections.shuffle(trainList);
			double loss = 0;
			for (int i = 0; i < trainList.size(); i++) {

				String targetPOI = trainList.get(i).split(",")[0]; // l
				String contextCategory = trainList.get(i).split(",")[2];
				String contextPOI = trainList.get(i).split(",")[3];

				// positive target
				double[] postiveTargetPOIVec = targetPOIVecMap.get(targetPOI); // V(l)

				double[] gradientVl = new double[postiveTargetPOIVec.length];

				ArrayList<String> negativePOIList = new ArrayList<String>();
				ArrayList<double[]> negativePOIGradientList = new ArrayList<double[]>();
				ArrayList<String> contextPOIList = new ArrayList<String>();
				ArrayList<double[]> contextPOIGradientList = new ArrayList<double[]>();
				ArrayList<String> negativeCategoryList = new ArrayList<String>();
				ArrayList<double[]> negativeCategoryGradientList = new ArrayList<double[]>();
				ArrayList<String> contextCategoryList = new ArrayList<String>();
				ArrayList<double[]> contextCategoryGradientList = new ArrayList<double[]>();

				// Jl
				String[] contextPOITemp = contextPOI.split("#");
				HashSet<String> contextPOISet = new HashSet<String>();
				for (int j = 0; j < contextPOITemp.length; j++) {
					contextPOISet.add(contextPOITemp[j]);
				}

				for (int j = 0; j < contextPOITemp.length; j++) {
					String contextpoi = contextPOITemp[j];
					contextPOIList.add(contextpoi);
					double[] contextPOIVec = contextPOIVecMap.get(contextpoi);// V(l_x)

					double[] gradientVlx = new double[gradientVl.length];

					double VlxVl = 0;
					for (int m = 0; m < contextPOIVec.length; m++) {
						VlxVl += contextPOIVec[m] * postiveTargetPOIVec[m];
					}
					double q1 = getSigmoid(VlxVl);
					loss += -Math.log(q1);

					for (int m = 0; m < gradientVlx.length; m++) {
						gradientVlx[m] = -(1 - q1) * postiveTargetPOIVec[m] * weight;
						gradientVl[m] += -(1 - q1) * contextPOIVec[m] * weight;
					}
					contextPOIGradientList.add(gradientVlx);

					String negativePOIs = getNegative(negativeNum, contextPOISet, targetPOIFrequencyList);
					String[] negativePOITemp = negativePOIs.split("#");
					// negative targets

					for (int k = 0; k < negativePOITemp.length; k++) {
						String negativePOI = negativePOITemp[k];
						negativePOIList.add(negativePOI);

						double[] negativePOIVec = contextPOIVecMap.get(negativePOI); // V(l_e)
						double VleVl = 0;
						for (int m = 0; m < negativePOIVec.length; m++) {
							VleVl += negativePOIVec[m] * postiveTargetPOIVec[m];
						}
						double q2 = getSigmoid(-VleVl);

						double[] gradientVle = new double[gradientVl.length];
						for (int m = 0; m < gradientVle.length; m++) {
							gradientVle[m] = (1 - q2) * postiveTargetPOIVec[m] * weight;
							gradientVl[m] += (1 - q2) * negativePOIVec[m] * weight;
						}
						negativePOIGradientList.add(gradientVle);
						loss += -Math.log(q2);
					}
				}

				// Jc
				String[] contextCategoryTemp = contextCategory.split("#");
				HashSet<String> contextCategorySet = new HashSet<String>();// cx
				for (int j = 0; j < contextCategoryTemp.length; j++) {
					if (!contextCategoryTemp[j].equals("NULL")) {
						contextCategorySet.add(contextCategoryTemp[j]);
					}
				}

				for (int j = 0; j < contextCategoryTemp.length; j++) {
					if (!contextCategoryTemp[j].equals("NULL")) {

						String category = contextCategoryTemp[j];
						contextCategoryList.add(category);
						double[] contextCategoryVec = contextCategoryVecMap.get(category);// V(c_x)

						double VcxVl = 0;
						for (int m = 0; m < postiveTargetPOIVec.length; m++) {
							VcxVl += contextCategoryVec[m] * postiveTargetPOIVec[m];
						}
						double q3 = getSigmoid(VcxVl);
						loss += -Math.log(q3);

						double[] gradientVcx = new double[gradientVl.length];

						for (int m = 0; m < postiveTargetPOIVec.length; m++) {
							gradientVcx[m] = -(1 - q3) * postiveTargetPOIVec[m] * weight;
							gradientVl[m] += -(1 - q3) * contextCategoryVec[m] * weight;
						}
						contextCategoryGradientList.add(gradientVcx);

						String negativeCategories = getNegative(negativeNum, contextCategorySet,
								targetCategoryFrequencyList);
						String[] negativeCategoryTemp = negativeCategories.split("#");
						// negative targets

						for (int k = 0; k < negativeCategoryTemp.length; k++) {
							String negativeCategory = negativeCategoryTemp[k];
							negativeCategoryList.add(negativeCategory);

							double[] negativeCategoryVec = contextCategoryVecMap.get(negativeCategory); // V(c_e)
							double VceVl = 0;
							for (int m = 0; m < postiveTargetPOIVec.length; m++) {
								VceVl += negativeCategoryVec[m] * postiveTargetPOIVec[m];
							}
							double q4 = getSigmoid(-VceVl);
							double[] gradientVce = new double[gradientVl.length];
							for (int m = 0; m < gradientVce.length; m++) {
								gradientVce[m] = (1 - q4) * postiveTargetPOIVec[m] * weight;
								gradientVl[m] += (1 - q4) * negativeCategoryVec[m] * weight;
							}
							negativeCategoryGradientList.add(gradientVce);
							loss += -Math.log(q4);
						}
					}
				}

				// Js
				for (int j = 0; j < contextCategoryTemp.length; j++) {
					if (!contextCategoryTemp[j].equals("NULL")) {
						String contextpoi = contextPOITemp[j];
						String category = contextCategoryTemp[j];

						ArrayList<String> negativeSampleCategoryList = getNegativeMultiple(category, categoryList,
								negativeCategoryNum);

						double[] contextPOIVec = contextPOIVecMap.get(contextpoi); // v(l_w)
						double[] contextCategoryVec = contextCategoryVecMap.get(category); // v(c_w)

						for (int k = 0; k > negativeSampleCategoryList.size(); k++) {
							String negativeCategory = negativeSampleCategoryList.get(k);
							double[] negativeCategoryVec = contextCategoryVecMap.get(negativeCategory); // v(c_e)
							double Dlce = getEuclideanDistance(negativeCategoryVec, contextPOIVec);
							double Dlc = getEuclideanDistance(contextCategoryVec, contextPOIVec);
							double sigmoidZ = getSigmoid(Dlce - Dlc);

							double[] gradientVlw = new double[gradientVl.length];
							double[] gradientVcw = new double[gradientVl.length];
							double[] gradientVce = new double[gradientVl.length];
							for (int m = 0; m < contextPOIVec.length; m++) {
								gradientVcw[m] = -2 * (1 - sigmoidZ) * (contextPOIVec[m] - contextCategoryVec[m])
										* (1 - weight);
								gradientVlw[m] = -2 * (1 - sigmoidZ) * (contextCategoryVec[m] - negativeCategoryVec[m])
										* (1 - weight);
								gradientVce[m] = 2 * (1 - sigmoidZ) * (contextPOIVec[m] - negativeCategoryVec[m])
										* (1 - weight);
							}
							negativeCategoryList.add(negativeCategory);
							negativeCategoryGradientList.add(gradientVce);
							contextCategoryList.add(category);
							contextCategoryGradientList.add(gradientVcw);
							contextPOIList.add(contextpoi);
							contextPOIGradientList.add(gradientVlw);

							loss += -Math.log(sigmoidZ);
						}

					}
				}

				// update gradient
				for (int j = 0; j < contextPOIList.size(); j++) {
					String poi = contextPOIList.get(j);
					double[] gradient = contextPOIGradientList.get(j);

					double[] vec = contextPOIVecMap.get(poi);
					for (int m = 0; m < vec.length; m++) {
						vec[m] -= alpha * (gradient[m] + lambda * vec[m]);
					}
					contextPOIVecMap.put(poi, vec);
				}

				for (int j = 0; j < negativePOIList.size(); j++) {
					String poi = negativePOIList.get(j);
					double[] gradient = negativePOIGradientList.get(j);

					double[] vec = contextPOIVecMap.get(poi);
					for (int m = 0; m < vec.length; m++) {
						vec[m] -= alpha * (gradient[m] + lambda * vec[m]);
					}
					contextPOIVecMap.put(poi, vec);
				}

				for (int j = 0; j < contextCategoryList.size(); j++) {
					String category = contextCategoryList.get(j);
					double[] gradient = contextCategoryGradientList.get(j);

					double[] vec = contextCategoryVecMap.get(category);
					for (int m = 0; m < vec.length; m++) {
						vec[m] -= alpha * (gradient[m] + lambda * vec[m]);
					}
					contextCategoryVecMap.put(category, vec);
				}

				for (int j = 0; j < negativeCategoryList.size(); j++) {
					String category = negativeCategoryList.get(j);
					double[] gradient = negativeCategoryGradientList.get(j);

					double[] vec = contextCategoryVecMap.get(category);
					for (int m = 0; m < vec.length; m++) {
						vec[m] -= alpha * (gradient[m] + lambda * vec[m]);
					}
					contextCategoryVecMap.put(category, vec);
				}

				for (int m = 0; m < postiveTargetPOIVec.length; m++) {
					postiveTargetPOIVec[m] -= alpha * (gradientVl[m] + lambda * postiveTargetPOIVec[m]);
				}
				targetPOIVecMap.put(targetPOI, postiveTargetPOIVec);

			}

			System.out.println("loss=" + loss);
			itrNum++;

			if (itrNum > 30) {
				break;
			}

		}
	}

	private double getEuclideanDistance(double[] negativeCategoryVec1, double[] postiveTargetPOIVec) {
		double dis = 0;

		for (int i = 0; i < negativeCategoryVec1.length; i++) {
			dis += (negativeCategoryVec1[i] - postiveTargetPOIVec[i])
					* (negativeCategoryVec1[i] - postiveTargetPOIVec[i]);
		}
		return dis;
	}

	// sample negative category for a query category
	private String getNegative(int sampleNum, HashSet<String> querySet, ArrayList<String> targetFrequencyList) {
		// negative target category
		String negative = "";
		ArrayList<String> negativeCategoryList = new ArrayList<String>();
		int sampleCount = 0, count = 0;
		while (true) {
			count++;
			int index = (int) Math.min(targetFrequencyList.size() - 1, Math.random() * targetFrequencyList.size());
			String sampleTargetCategory = targetFrequencyList.get(index);
			if (!querySet.contains(sampleTargetCategory)) {
				negativeCategoryList.add(sampleTargetCategory);
				sampleCount++;
			}
			// if sample 50 times do not enough negative samples, stop
			if (sampleCount == sampleNum || count > 50) {
				break;
			}
		}

		for (int i = 0; i < negativeCategoryList.size() - 1; i++) {
			negative += negativeCategoryList.get(i) + "#";
		}
		negative += negativeCategoryList.get(negativeCategoryList.size() - 1);
		return negative;
	}

	// sample negative category for a query category, not frequency category list
	private String getNegativeOne(String query, ArrayList<String> categoryList) {
		// negative target category
		String negative = "";
		int sampleCount = 0;
		while (true) {
			int index = (int) Math.min(categoryList.size() - 1, Math.random() * categoryList.size());
			String sampleTargetCategory = categoryList.get(index);
			if (!query.equals(sampleTargetCategory)) {
				negative = sampleTargetCategory;
				sampleCount++;
			}
			// if sample 50 times do not enough negative samples, stop
			if (sampleCount == 1) {
				break;
			}
		}
		return negative;
	}

	private ArrayList<String> getNegativeMultiple(String query, ArrayList<String> categoryList, int negativeNum) {
		// negative target category
		ArrayList<String> negative = new ArrayList<String>();
		int sampleCount = 0;
		while (true) {
			int index = (int) Math.min(categoryList.size() - 1, Math.random() * categoryList.size());
			String sampleTargetCategory = categoryList.get(index);
			if (!query.equals(sampleTargetCategory)) {
				negative.add(sampleTargetCategory);
				sampleCount++;
			}
			// if sample 50 times do not enough negative samples, stop
			if (sampleCount == negativeNum) {
				break;
			}
		}
		return negative;
	}

	private void getTrainList(String trainPath, ArrayList<String> trainList, int contextWindowSize) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(trainPath));
		while (true) {
			String line = br.readLine();
			if (line == null)
				break;
			String[] temp = line.split(",");
			for (int i = 0; i < temp.length; i++) {
				String targetPOI = temp[i].split("#")[0];
				String targetCategory = temp[i].split("#")[1];
				// context category
				ArrayList<String> contextCategortList = new ArrayList<String>();
				ArrayList<String> contextPOIList = new ArrayList<String>();
				for (int j = i - 1; j >= Math.max(0, i - contextWindowSize); j--) {
					String poi = temp[j].split("#")[0];
					String category = temp[j].split("#")[1];
					contextCategortList.add(category);
					contextPOIList.add(poi);
				}
				for (int j = i + 1; j <= Math.min(temp.length - 1, i + contextWindowSize); j++) {
					String poi = temp[j].split("#")[0];
					String category = temp[j].split("#")[1];
					contextCategortList.add(category);
					contextPOIList.add(poi);
				}

				String record = targetPOI + "," + targetCategory + ",";

				if (contextCategortList.size() > 0) {
					for (int j = 0; j < contextCategortList.size() - 1; j++) {
						record += contextCategortList.get(j) + "#";
					}
					record += contextCategortList.get(contextCategortList.size() - 1) + ",";

					for (int j = 0; j < contextPOIList.size() - 1; j++) {
						record += contextPOIList.get(j) + "#";
					}
					record += contextPOIList.get(contextPOIList.size() - 1);

					trainList.add(record);
				}

			}
		}
		br.close();
	}

	// compute category list according to the frequency
	private void computeTargetFrequencyList(ArrayList<String> trainList, ArrayList<String> poiFrequencyList,
			ArrayList<String> categoryFrequencyList, ArrayList<String> categoryList, ArrayList<String> poiList)
			throws IOException {
		HashMap<String, Integer> candidatePOICountMap = new HashMap<String, Integer>();
		HashMap<String, Integer> candidateCategoryCountMap = new HashMap<String, Integer>();
		for (int i = 0; i < trainList.size(); i++) {
			String poi = trainList.get(i).split(",")[0];
			String category = trainList.get(i).split(",")[1];
			if (candidatePOICountMap.containsKey(poi)) {
				candidatePOICountMap.put(poi, candidatePOICountMap.get(poi) + 1);
			} else {
				candidatePOICountMap.put(poi, 1);
			}

			if (candidateCategoryCountMap.containsKey(category)) {
				candidateCategoryCountMap.put(category, candidateCategoryCountMap.get(category) + 1);
			} else {
				candidateCategoryCountMap.put(category, 1);
			}
		}

		// normalization
		Set<String> poiSet = candidatePOICountMap.keySet();
		Iterator<String> itr = poiSet.iterator();
		int minPOI = 10000;
		while (itr.hasNext()) {
			String poi = itr.next();
			int count = candidatePOICountMap.get(poi);
			if (count < minPOI) {
				minPOI = count;
			}
		}

		Set<String> categorySet = candidateCategoryCountMap.keySet();
		Iterator<String> itr1 = categorySet.iterator();
		int minCategory = 10000;
		while (itr1.hasNext()) {
			String category = itr1.next();
			int count = candidateCategoryCountMap.get(category);
			if (count < minCategory) {
				minCategory = count;
			}
		}

		// num of category is related to the frequency's 3/4 cifang
		itr = poiSet.iterator();
		while (itr.hasNext()) {
			String poi = itr.next();
			int count = candidatePOICountMap.get(poi);
			int newcount = (int) Math.pow(count / (minPOI + 0.0), 3 / 4);
			for (int i = 0; i < newcount; i++) {
				poiFrequencyList.add(poi);
			}
			poiList.add(poi);
		}

		itr = categorySet.iterator();
		while (itr.hasNext()) {
			String category = itr.next();
			int count = candidateCategoryCountMap.get(category);
			int newcount = (int) Math.pow(count / (minCategory + 0.0), 3 / 4);
			for (int i = 0; i < newcount; i++) {
				categoryFrequencyList.add(category);
			}
			categoryList.add(category);
		}
	}

	private void saveModel(HashMap<String, double[]> contextCategoryVecMap1, String path) throws IOException {
		PrintWriter pw = new PrintWriter(new FileWriter(path));
		Set<String> categorySet = contextCategoryVecMap1.keySet();
		Iterator<String> itr = categorySet.iterator();
		while (itr.hasNext()) {
			String category = itr.next();
			pw.print(category);
			double[] vec = contextCategoryVecMap1.get(category);
			for (int i = 0; i < vec.length; i++) {
				pw.print("," + vec[i]);
			}
			pw.println();
		}
		pw.flush();
		pw.close();
	}

	private double getSigmoid(double z) {
		double sigmoidZ = 0;
		// sigmoid
		if (z <= -MAX_EXP)
			sigmoidZ = 0.000001;
		else if (z >= MAX_EXP)
			sigmoidZ = 0.999999;
		else
			sigmoidZ = expTable[(int) ((z + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		return sigmoidZ;
	}

	/**
	 * Precompute the exp() table sigmoid function jinsi jisuan f(x) = x / (x + 1)
	 */
	private void createExpTable() {
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
			expTable[i] = expTable[i] / (expTable[i] + 1);
		}
	}
}

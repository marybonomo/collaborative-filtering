package algorithm;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.mllib.recommendation.Rating;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

import algorithm.JavaALS.ParseRating;
import scala.Tuple2;

public class JavaALSExample{  
		
    public static class Schema implements Serializable {
       
        public String getID() {
            return id;
        }
        public void setId(String id) {
            this.id = id;
        }
        public String getMovie() {
            return movie;
        }
        public void setMovie(String movie) {
            this.movie = movie;
        }
        public String getScore() {
            return score;
        }
        public void setScore(String score) {
            this.score = score;
        }
        public String getTimestamp() {
            return timestamp;
        }
        public void setTimestamp(String timestamp) {
            this.timestamp = timestamp;
        }  
        
        //instance variables
        private String id;
        private String movie;
        private String score;
        private String timestamp;
    }

    public static void main(String[] args) {
    	 SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("ALS");
         JavaSparkContext sc=new JavaSparkContext(sparkConf);
    	
    	SparkSession spark=SparkSession
                .builder()
                .appName("ALSExample")
                .master("local[2]")
                //.config("spark.executor.memory", "8g")
                //.config("spark.executor.cores", 4)
                .getOrCreate();
    	sc.setLogLevel("ERROR");
        SQLContext sqlcontext= new SQLContext(sc);

        String path="C:/Users/Mary/eclipse-workspace/CollaborativeFiltering/src/main/resources/ratings_without_header.csv";

                
        //-------------------------------1.0 Prepare DataFrame---------------- ------------
        //..javaRDD() function converts DataFrame to RDD
        //Then map each row of RDD String->Rating
                
        JavaRDD<Schema> schemaRdd = sc.textFile(path).map(
                        new Function<String, Schema>() {
                            public Schema call(String line) throws Exception {
                                String[] tokens=line.split(",");
                                Schema schema = new Schema();
                                schema.setId(tokens[0]);
                                schema.setMovie(tokens[1]);
                                schema.setScore(tokens[2]);
                                schema.setTimestamp(tokens[3]);
                                return schema;
                            }
                        });       
        
        //dataset originario
        Dataset<Row> df = sqlcontext.createDataFrame(schemaRdd, Schema.class);
        df.createOrReplaceTempView("rating");        
        Dataset<Row> sqlDF = spark.sql("SELECT * FROM rating");
        //sqlDF.show(10); // prova stampa tabella
        
        //dataset con 3 colonne
        Dataset<Row> db = sqlcontext.createDataFrame(schemaRdd, Schema.class);
        db.createOrReplaceTempView("rating");        
        Dataset<Row> sqlDb = spark.sql("SELECT INT(ID), INT(movie), INT(score) FROM rating");
        //sqlDb.show(10); // prova stampa tabella
        
        
        //training set and test set
        double[] weights=new double[] {0.8,0.2};
        long seed=1234;
        Dataset [] split=sqlDb.randomSplit(weights, seed);
        Dataset training=split[0];
        Dataset test=split[1];         
        List RMSE=new ArrayList(); //
        //------------------------------2.0 ALS algorithm and training data set to generate recommendation model --------- ----
        for(int rank=1;rank<20;rank++) ////20 cicli
        {
            //algoritmo
            ALS als=new ALS()
                    .setMaxIter(15)////The maximum number of iterations, the setting is too large, java.lang.StackOverflowError occurs
                    .setRegParam(0.05)
                    .setAlpha(1.0)
                    .setImplicitPrefs(false)
                    .setNonnegative(false)
                    .setUserCol("ID")               
                    .setRank(rank)
                    .setItemCol("movie")
                    .setRatingCol("score");
                                     
            		
            //Training model
            ALSModel model=als.fit(training);
                        
            //---------------------------3.0 Model Evaluation: Calculate RMSE, Root Mean Square Error------------ ---------
           
            Dataset predictions=model.transform(test);
            //predictions.na().drop();
            predictions.show();
            
            
            RegressionEvaluator evaluator=new RegressionEvaluator()
                    .setMetricName("rmse")
                    .setLabelCol("score")
                    .setPredictionCol("prediction");
               		
            Double rmse=evaluator.evaluate(predictions.na().drop());
            
            RMSE.add(rmse);
            System.out.println("Rank =" + rank+"  RMSErr = " + rmse+"\n");
            
           
            ///output all result
            double cont=(float) 0.0;
            for (int j=0;j<RMSE.size();j++) {
            	Double lambda=(j*5+1)*0.01; 
            	System.out.println("regParam="+lambda+" RMSE= "+ RMSE.get(j)+"\n");
            	
            }
       
         

        }       
    }
}
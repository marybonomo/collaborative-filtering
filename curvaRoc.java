package algorithm;
/**
 * Curva.java il programma per generare la curva di Roc
 * il metodo usa un file dove sono memorizzati i valori di pvalue
 *
 * @author Mary Bonomo
 */

import be.cylab.java.roc.Roc;
import be.cylab.java.roc.RocCoordinates;
import be.cylab.java.roc.Utils;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
//import be.cylab.java.roc.PRCurve;
import org.apache.spark.mllib.evaluation.MultilabelMetrics;
import org.apache.spark.rdd.RDD;

import java.util.Arrays;
import java.util.List;

public class curvaRoc {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SparkConsumer").setMaster("local[*]");
        @SuppressWarnings("resource")
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        Roc roc = new Roc("src/main/resources/perAUC1.csv");
        //AucDef0.csv dataset 20% che contiene anche le combinazioni dell'80 : 0.9921
        //AucDef.csv dataset 20% meno l'80: 0.7037
        //score30.csv dataset 30% meno il 70: 0.6799
        //score10.csv dataset 10% meno il 90: 0.6601
        //prova intermedia con label.csv
        //prova def collab filtering perAUC1.csv
        
        //AUC computation and printing
        System.out.println(roc.computeAUC());
        
      
        //Roc points computation
        List<RocCoordinates> roc_coordinates = roc.computeRocPointsAndGenerateCurve("src/main/resources/png/curva.png");

        //Save RocCoordinates in a CSV file
        Utils.storeRocCoordinatesInCSVFile(roc_coordinates, "src/main/resources/curva/curva.csv");
          
        
    }

}
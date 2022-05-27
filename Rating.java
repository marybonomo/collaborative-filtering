package algorithm;

import java.io.IOException;
import java.io.Serializable;
import org.apache.spark.api.java.JavaRDD;
//import org.apache.spark.ml.evaluation.RegressionEvaluator;
//import org.apache.spark.ml.recommendation.ALS;
//import org.apache.spark.ml.recommendation.ALSModel;

public class Rating implements Serializable {
 
private int userId;
  private int movieId;
  private float rating;
  private long timestamp;

  public Rating(int x, int y, double rating2) {}

  public Rating(int userId, int movieId, float rating, long timestamp) {
    this.userId = userId;
    this.movieId = movieId;
    this.rating = rating;
    this.timestamp = timestamp;
  }

  public int getUserId() {
    return userId;
  }

  public int getMovieId() {
    return movieId;
  }

  public float getRating() {
    return rating;
  }

  public long getTimestamp() {
    return timestamp;
  }

  public static Rating parseRating(String str) {
    String[] fields = str.split("::");
    if (fields.length != 4) {
      throw new IllegalArgumentException("Each line must contain 4 fields");
    }
    int userId = Integer.parseInt(fields[0]);
    int movieId = Integer.parseInt(fields[1]);
    float rating = Float.parseFloat(fields[2]);
    long timestamp = Long.parseLong(fields[3]);
    return new Rating(userId, movieId, rating, timestamp);
  }
}
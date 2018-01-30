import java.lang.*;

class Racer implements Runnable{
    private int DIST=100;    
    public static String winner;

    public void race(){
        for(int dist=1;dist<=DIST;++dist){
            System.out.println("Distance by "+Thread.currentThread().getName()+" is "+dist);

            //Check common;;;
            if(won(dist))
                break;
        }
    }
    private boolean won(int dist){
        boolean won=false;
        if(Racer.winner==null && dist==DIST){
            Racer.winner=Thread.currentThread().getName();
            System.out.println(Racer.winner + "won the race");
            won=true;
        }
        if(Racer.winner!=null)
            won = true;
        
        return won;
    }

    public void run(){
        this.race();
    }
}

public class hareTurtle{
    public static void main(String args[]){
        Racer racer=new Racer();
        Thread turtleThread =  new Thread(racer,"Turtle");
        Thread hareThread  =  new Thread(racer,"Hare");
        
        hareThread.start();
        turtleThread.start();
    }
}
import java.lang.*;
class Print implements Runnable{
    //extends Thread{
    //Nothing else implemented after inheritence.
    //Bad OO practice.So runnable used :P
    private String data;
    private int delay;
    public Print(String data,int delay){
        this.data=data;
        this.delay=delay;
    }
    public void run(){
        for(int i=0;i<5;++i){
            System.out.println((i)*delay+" ms--> "+data);
            try        
                {
                    Thread.sleep(delay);
                } 
                catch(InterruptedException ex) 
                {
                    Thread.currentThread().interrupt();
                    //throw new RuntimeException(e);
                }
        }
    }
}
public class basicThread{
    public static void main(String args[]){
        System.out.println("This shows MultiThreading in Java");
        Print taskA = new Print("Praveen",1000);
        Print taskB = new Print("Gupta",300);

        //Directs thread to fin run method in object taskA
        Thread A = new Thread(taskA);
        Thread B = new Thread(taskB);
        
        A.start();
        B.start();
    }
}
/*
This shows MultiThreading in Java
0 ms--> Praveen
0 ms--> Gupta
300 ms--> Gupta
600 ms--> Gupta
900 ms--> Gupta
1000 ms--> Praveen
1200 ms--> Gupta
2000 ms--> Praveen
3000 ms--> Praveen
4000 ms--> Praveen
*/
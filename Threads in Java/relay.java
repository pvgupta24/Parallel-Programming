import java.lang.*;
import java.util.concurrent.CountDownLatch;

class P1 implements Runnable{
    private final CountDownLatch countDownLatch;
    private final String s;
    private final int DIST=5;
    public P1(CountDownLatch c, String s){
        this.countDownLatch=c;
        this.s=s;
    }

    @Override
    public void run(){
        for(int i=0;i<DIST;++i){
            System.out.println(s+" -> " +i);
        }
        countDownLatch.countDown();
    }
}
class P2 implements Runnable{
    private final CountDownLatch countDownLatch;
    private final String s;
    private final int DIST=5;
    
    public P2(CountDownLatch c, String s){
        this.countDownLatch=c;
        this.s=s;
    }
    @Override
    public void run(){
        try {
            countDownLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        for(int i=0;i<DIST;++i)
            System.out.println(s+" -> "+i);
    }
}


public class relay{
    public static void main(String args[]){
        System.out.println("Relay Race");
        //Thread[][] teams = new Thread[5][2];//5 teams i.e thread groups of 2 player i.e thread each;
        Thread[] ind = new Thread[2];
        Thread[] aus = new Thread[2];        

        //Player B should wait only for Player1
        CountDownLatch c1 = new CountDownLatch(1);
        CountDownLatch c2 = new CountDownLatch(1);

        ind[0]=new Thread(new P1(c1,"INDIA-A"));
        ind[1]=new Thread(new P2(c1,"INDIA-B"));

        aus[0]=new Thread(new P1(c2,"Aussies-A"));
        aus[1]=new Thread(new P2(c2,"Aussies-B"));  

        ind[0].start();
        ind[1].start();
        aus[0].start();
        aus[1].start();
        
    }
}
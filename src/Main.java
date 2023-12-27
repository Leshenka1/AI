import java.util.*;

public class Main {
    public static void main(String[] args) {
        // Define some rules
        Condition condition1_1 = new Condition("engine", "diesel");
        Condition condition1_2 = new Condition("transmission", "full");
        Condition conclusion1 = new Condition("type", "jeep");

        Condition condition2_1 = new Condition("engine", "petrol");
        Condition condition2_2 = new Condition("transmission", "back");
        Condition condition2_3 = new Condition("capacity", "2");
        Condition conclusion2 = new Condition("type", "roadster");

        Condition condition3_1 = new Condition("engine", "diesel");
        Condition condition3_2 = new Condition("transmission", "front");
        Condition condition3_3 = new Condition("capacity", "7");
        Condition conclusion3 = new Condition("type", "minivan");

        Condition condition4_1 = new Condition("engine", "petrol");
        Condition condition4_2 = new Condition("transmission", "front");
        Condition condition4_3 = new Condition("capacity", "5");
        Condition condition4_4 = new Condition("under gear", "no");
        Condition conclusion4 = new Condition("type", "sedan");

        Condition condition5_1 = new Condition("cylinder amount", "5");
        Condition condition5_2 = new Condition("efficiency", "yes");
        Condition conclusion5 = new Condition("engine", "diesel");

        Condition condition6_1 = new Condition("cylinder amount", "8");
        Condition condition6_2 = new Condition("efficiency", "no");
        Condition conclusion6 = new Condition("engine", "petrol");

        Condition condition7_1 = new Condition("spark-plug", "yes");
        Condition condition7_2 = new Condition("cylinder position", "w");
        Condition condition7_3 = new Condition("horse power", "high");
        Condition conclusion7 = new Condition("cylinder amount", "10");

        Condition condition8_1 = new Condition("spark-plug", "yes");
        Condition condition8_2 = new Condition("cylinder position", "w");
        Condition conclusion8 = new Condition("cylinder amount", "12");

        Condition condition9_1 = new Condition("spark-plug", "no");
        Condition condition9_2 = new Condition("cylinder position", "row");
        Condition conclusion9 = new Condition("cylinder amount", "5");

        Condition condition10_1 = new Condition("spark-plug", "yes");
        Condition condition10_2 = new Condition("cylinder position", "v");
        Condition condition10_3 = new Condition("horse power", "low");
        Condition conclusion10 = new Condition("cylinder amount", "8");

        Condition condition11_1 = new Condition("drive ability", "low");
        Condition condition11_2 = new Condition("crankshaft", "no");
        Condition conclusion11 = new Condition("transmission", "front");

        Condition condition12_1 = new Condition("drive ability", "low");
        Condition condition12_2 = new Condition("crankshaft", "yes");
        Condition conclusion12 = new Condition("transmission", "back");

        Condition condition13_1 = new Condition("drive ability", "high");
        Condition conclusion13 = new Condition("transmission", "full");

        Rule rule1 = new Rule(1, Arrays.asList(condition1_1, condition1_2), conclusion1);
        Rule rule2 = new Rule(2, Arrays.asList(condition2_1, condition2_2, condition2_3), conclusion2);
        Rule rule3 = new Rule(3, Arrays.asList(condition3_1, condition3_2, condition3_3), conclusion3);
        Rule rule4 = new Rule(4, Arrays.asList(condition4_1, condition4_2, condition4_3, condition4_4), conclusion4);
        Rule rule5 = new Rule(5, Arrays.asList(condition5_1, condition5_2), conclusion5);
        Rule rule6 = new Rule(6, Arrays.asList(condition6_1, condition6_2), conclusion6);
        Rule rule7 = new Rule(7, Arrays.asList(condition7_1, condition7_2, condition7_3), conclusion7);
        Rule rule8 = new Rule(8, Arrays.asList(condition8_1, condition8_2), conclusion8);
        Rule rule9 = new Rule(9, Arrays.asList(condition9_1, condition9_2), conclusion9);
        Rule rule10 = new Rule(10, Arrays.asList(condition10_1, condition10_2, condition10_3), conclusion10);
        Rule rule11 = new Rule(11, Arrays.asList(condition11_1, condition11_2), conclusion11);
        Rule rule12 = new Rule(12, Arrays.asList(condition12_1, condition12_2), conclusion12);
        Rule rule13 = new Rule(13, Arrays.asList(condition13_1), conclusion13);


        // Define the initial goals
        Scanner scanner = new Scanner(System.in);
        System.out.println("Please enter name for rule u want to find");
        String goal = scanner.nextLine();
        Goal initialGoal = new Goal(goal, null);

        // Initialize the backward chaining algorithm
        BackwardChaining backwardChaining = new BackwardChaining(Arrays.asList(rule1, rule2, rule3, rule4,
                rule5, rule6, rule7, rule8, rule9, rule10), initialGoal);

        // Run the algorithm
        backwardChaining.run();
    }
}
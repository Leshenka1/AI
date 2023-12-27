import java.util.*;

public class BackwardChaining {
    private Goal goal;
    private Stack<Goal> goalStack;
    private Map<String, String> context;
    private List<Rule> rules;
    private ArrayList<Integer> fakeRules;



    // Assume that the rules and initial goals are passed as arguments to the constructor
    public BackwardChaining(List<Rule> rules, Goal goal) {
        this.rules = new ArrayList<>(rules);
        this.goalStack = new Stack<>();
        this.goal = goal;
        this.goalStack.push(goal);
        this.context = new HashMap<>();
        this.fakeRules = new ArrayList<>();
    }

    public void run() {
        while (!goalStack.isEmpty()) {
            Goal currentGoal = goalStack.peek();

//            if (context.containsKey(currentGoal.getName())) {
//                currentGoal.setValue(context.get(currentGoal.getName()));
//            }

            if (currentGoal.getValue() != null) {
                this.goal.setValue(currentGoal.getValue());
                goalStack.pop();
                continue;
            }

            Rule rule = findRule(currentGoal.getName());
            if (rule != null) {
                Boolean haveAllConditions = true;
                Boolean addFakeRule = false;
                for (Condition condition : rule.getConditions()) {
                    if (!context.containsKey(condition.getName()) || !context.get(condition.getName()).equals(condition.getValue())) {
                        if(context.get(condition.getName()) != null && !context.get(condition.getName()).equals(condition.getValue())){
                            fakeRules.add(rule.getId());
                            addFakeRule = true;
                        }
                        else {
                            goalStack.push(new Goal(condition.getName(), null));
                            haveAllConditions = false;
                        }
                    }
                }
                if(haveAllConditions && !addFakeRule){
                    context.put(currentGoal.getName(), rule.getConclusion().getValue());
                    goalStack.pop();
                }
            } else {
                // If we reach here, it means we don't have a rule for the current goal
                // Ask the user for the value
                Scanner scanner = new Scanner(System.in);
                System.out.println("Please enter value for " + currentGoal.getName());
                String input = scanner.nextLine();
                context.put(currentGoal.getName(), input);
                goalStack.pop();
            }
        }

        // At this point, the context map should contain the values for all goals
        System.out.println("Context: " + context.get(goal.getName()));
    }


    private Rule findRule(String goalName) {
        for (Rule rule : rules) {
            if (rule.getConclusion().getName().equals(goalName) && !fakeRules.contains(rule.getId())) {
                return rule;
            }
        }
        System.out.println("Sorry, we can't find rule with goal " + goalName);
        return null;
    }
//    private Rule findFakeRule(Integer id) {
//        for (Rule rule : rules) {
//            if (rule.getConclusion().getName().equals(goalName) && !fakeRules.contains(rule.getId())) {
//                return rule;
//            }
//        }
//        System.out.println("Sorry, we can't find rule with goal " + goalName);
//        return null;
//    }
}

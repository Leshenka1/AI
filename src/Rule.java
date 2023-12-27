import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Rule {
    private Integer id;
    private List<Condition> conditions;
    private Condition conclusion;

    public Rule(Integer id, List<Condition> conditions, Condition conclusion) {
        this.id = id;
        this.conditions = new ArrayList<>(conditions);
        this.conclusion = conclusion;
    }


    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public List<Condition> getConditions() {
        return conditions;
    }

    public void setConditions(List<Condition> conditions) {
        this.conditions = conditions;
    }

    public Condition getConclusion() {
        return conclusion;
    }

    public void setConclusion(Condition conclusion) {
        this.conclusion = conclusion;
    }



}
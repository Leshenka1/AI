public class Goal {
    private String value;
    private String name;


    public void setName(String name) {
        this.name = name;
    }
    public void setValue(String value) {
        this.value = value;
    }


    public String getValue() {
        return value;
    }

    public Goal(String name, String value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

}
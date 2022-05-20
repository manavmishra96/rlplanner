(define (domain boxworld)
    
    (:predicates

	    (agent-at ?x ?y)
	    (agent-key ?key)
	    (agent-achieved-gem )

	    (unlocked-box ?x ?y)
	    (locked-box ?x ?y)
	    (opens-by ?key ?x ?y)   
	    (contains-key ?key ?x ?y)  
	    (contains-gem ?x ?y) 
    )
    
    (:action move
        :parameters (?x1 ?y1 ?x2 ?y2)

        :precondition 
	        (and 
                (agent-at ?x1 ?y1)
                (not (agent-at ?x2 ?y2))
	        )

        :effect 
            (and 
                (agent-at ?x2 ?y2)
                (not (agent-at ?x1 ?y1))
            )
    )

    (:action unlock-box
        :parameters (?key ?x ?y)

        :precondition 
            (and 
                (agent-at ?x ?y)
		        (locked-box ?x ?y)
		        (agent-key ?key)
		        (opens-by ?key ?x ?y)
	        )

        :effect 
	        (and 
		        (not (locked-box ?x ?y))
		        (unlocked-box ?x ?y)
            )
    )
    
    (:action pickup-key
        :parameters (?key  ?x ?y)

        :precondition 
            (and 
                (agent-at ?x ?y)
		        (unlocked-box ?x ?y)
		        (contains-key ?key ?x ?y)
	        )

        :effect 
	        (and
		        (agent-key ?key)
		        (not (contains-key ?key ?x ?y))
	        )
    )

    (:action pickup-gem
        :parameters (?x ?y)

        :precondition 
            (and 
                (agent-at ?x ?y)
		        (unlocked-box ?x ?y)
		        (contains-gem ?x ?y)
	    )

        :effect 
	        (and
		        (agent-achieved-gem)
		        (not (contains-gem ?x ?y))
	        )
    )
)
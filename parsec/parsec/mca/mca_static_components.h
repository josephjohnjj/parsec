#ifndef _MCA_STATIC_COMPNENTS_H
#define _MCA_STATIC_COMPNENTS_H

#ifndef MCA_REPOSITORY_C
#error This file must be included once only, and by mca_repository.c only
#endif

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/utils/mca_param.h"
#include "parsec/utils/output.h"
#include <assert.h>

#define MCA_NB_STATIC_COMPONENTS 14

mca_base_component_t *pins_iterators_checker_static_component(void);
mca_base_component_t *pins_print_steals_static_component(void);
mca_base_component_t *pins_ptg_to_dtd_static_component(void);
mca_base_component_t *pins_task_granularity_static_component(void);
mca_base_component_t *sched_ap_static_component(void);
mca_base_component_t *sched_gd_static_component(void);
mca_base_component_t *sched_ip_static_component(void);
mca_base_component_t *sched_lfq_static_component(void);
mca_base_component_t *sched_lhq_static_component(void);
mca_base_component_t *sched_ll_static_component(void);
mca_base_component_t *sched_ltq_static_component(void);
mca_base_component_t *sched_pbq_static_component(void);
mca_base_component_t *sched_rnd_static_component(void);
mca_base_component_t *sched_spq_static_component(void);

static mca_base_component_t *mca_static_components[MCA_NB_STATIC_COMPONENTS+1] = { NULL, };

static int add_static_component(mca_base_component_t *c, int p)
{
    if( NULL == c )
        return p;
    assert( p < MCA_NB_STATIC_COMPONENTS );    mca_static_components[p] = c;
    mca_static_components[p+1] = NULL;
    return p+1;
}

static void register_base_component(const char *cname)
{
    char *help, *ignored;
    int rc;

    rc = asprintf(&help, "Default selection set of components for the %s framework "
                  "(<not set> means use all components that can be found)", cname);
    rc = parsec_mca_param_reg_string_name("mca", cname,
                                          help,
                                          false, false,
                                          NULL, &ignored);
    if( 0 < rc ) {  /* parameter succesfully registered */
        /* Create a synonym to facilitate the MCA params */
        (void)parsec_mca_param_reg_syn_name(rc, NULL, cname, false);
    }
    free(help);
    rc = asprintf(&help, "Verbosity level for the %s framework (default: 0). "
                  "Valid values: -1:\"none\", 0:\"error\", 10:\"component\", 20:\"warn\", "
                  "40:\"info\", 60:\"trace]\", 80:\"debug\", 100:\"max]\", 0 - 100", cname);
    parsec_mca_param_reg_int_name(cname, "verbose",
                                  help, false, false,
                                  0, (int*)&ignored);
    free(help);
    (void)ignored;
    (void)rc;
}

static void mca_static_components_init(void)
{
    static int mca_static_components_inited = 0;
    int p = 0;
    if (mca_static_components_inited) {
        return;
    }
    mca_static_components_inited = 1;

      register_base_component("device");
    p = add_static_component(pins_iterators_checker_static_component(), p);
    p = add_static_component(pins_print_steals_static_component(), p);
    p = add_static_component(pins_ptg_to_dtd_static_component(), p);
    p = add_static_component(pins_task_granularity_static_component(), p);  register_base_component("pins");
    p = add_static_component(sched_ap_static_component(), p);
    p = add_static_component(sched_gd_static_component(), p);
    p = add_static_component(sched_ip_static_component(), p);
    p = add_static_component(sched_lfq_static_component(), p);
    p = add_static_component(sched_lhq_static_component(), p);
    p = add_static_component(sched_ll_static_component(), p);
    p = add_static_component(sched_ltq_static_component(), p);
    p = add_static_component(sched_pbq_static_component(), p);
    p = add_static_component(sched_rnd_static_component(), p);
    p = add_static_component(sched_spq_static_component(), p);  register_base_component("sched");
}

#endif /* _MCA_STATIC_COMPNENTS_H */
